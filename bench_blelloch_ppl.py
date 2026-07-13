import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import time
import sys
import os

sys.path.insert(0, '/tmp/opencode/mamba-forensic')
from mamba_ssm.ops.selective_scan_blelloch import blelloch_ssm_fwd, ref_ssm_scan


def monkey_patch_mamba2(model):
    """Replace the chunked scan in Mamba2 with Blelloch scan.
    
    The Mamba2 model uses selective_scan_fn in MambaInnerFn.forward.
    We monkey-patch the selective_scan_fn to use our Blelloch scan.
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    
    original_fn = selective_scan_fn
    
    def blelloch_scan_wrapper(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                               delta_softplus=False, return_last_state=False):
        """Wrapper that converts Mamba1-style args to our Blelloch scan."""
        batch, dim, seqlen = u.shape
        dstate = A.shape[-1] * (1 if not A.is_complex() else 2)
        
        u_r = rearrange(u.float(), "b d l -> b l d")
        delta_r = rearrange(delta.float(), "b d l -> b l d")
        
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        
        if A.is_complex():
            B_c = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            C_c = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
            nheads = dim
            ngroups = B_c.shape[1] if B_c.dim() >= 3 else 1
            if not is_variable_B:
                B_exp = repeat(B_c, "d n -> b l d n", b=batch, l=1).expand(-1, seqlen, -1, -1)
            else:
                B_exp = B_c.permute(0, 2, 1, 3) if B_c.dim() == 4 else B_c.unsqueeze(1).expand(-1, nheads, -1, -1).permute(0, 2, 1, 3)
            if not is_variable_C:
                C_exp = repeat(C_c, "d n -> b l d n", b=batch, l=1).expand(-1, seqlen, -1, -1)
            else:
                C_exp = C_c.permute(0, 2, 1, 3) if C_c.dim() == 4 else C_c.unsqueeze(1).expand(-1, nheads, -1, -1).permute(0, 2, 1, 3)
            
            out = blelloch_ssm_fwd(u_r, delta_r, A, B_exp, C_exp,
                                    D=torch.view_as_real(D).sum(-1) if D is not None else None,
                                    z=rearrange(z, "b d l -> b l d") if z is not None else None,
                                    delta_bias=delta_bias,
                                    delta_softplus=delta_softplus,
                                    return_last_state=return_last_state)
            if return_last_state:
                out, last_state = out
            out = rearrange(out, "b l d -> b d l")
            if return_last_state:
                return out, last_state
            return out
        
        if A.dim() == 2:
            nheads = A.shape[0]
        else:
            nheads = dim
        
        if is_variable_B:
            if B.dim() == 3:
                B_reshaped = rearrange(B.float(), "b n l -> b l 1 n")
            else:
                B_reshaped = rearrange(B.float(), "b g n l -> b l g n")
        else:
            B_reshaped = repeat(B.float(), "d n -> b l d n", b=batch, l=seqlen)
            B_reshaped = rearrange(B_reshaped, "b l d n -> b l 1 n")
        
        if is_variable_C:
            if C.dim() == 3:
                C_reshaped = rearrange(C.float(), "b n l -> b l 1 n")
            else:
                C_reshaped = rearrange(C.float(), "b g n l -> b l g n")
        else:
            C_reshaped = repeat(C.float(), "d n -> b l d n", b=batch, l=seqlen)
            C_reshaped = rearrange(C_reshaped, "b l d n -> b l 1 n")
        
        A_reshaped = A.float()
        
        out = blelloch_ssm_fwd(
            rearrange(u.float(), "b d l -> b l d"),
            rearrange(delta.float(), "b d l -> b l d"),
            A_reshaped,
            B_reshaped,
            C_reshaped,
            D=rearrange(D.float(), "d -> d 1") if D is not None else None,
            z=rearrange(z.float(), "b d l -> b l d") if z is not None else None,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
        
        if return_last_state:
            out, last_state = out
            out = rearrange(out, "b l d -> b d l")
            return out, last_state
        out = rearrange(out, "b l d -> b d l")
        return out
    
    import mamba_ssm.ops.selective_scan_interface as ssi
    ssi.selective_scan_fn = blelloch_scan_wrapper
    print("  [patch] selective_scan_fn → Blelloch scan")
    return original_fn


def measure_ppl(model, tokenizer, dataset_name="wikitext-2", split="test", max_samples=50):
    """Measure perplexity on a dataset."""
    from transformers import AutoTokenizer
    
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  Could not load dataset: {e}")
        print("  Using synthetic data instead")
        return measure_ppl_synthetic(model)
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b")
    
    total_loss = 0.0
    total_tokens = 0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            text = example.get("text", "")
            if not text or len(text) < 10:
                continue
            
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = tokens["input_ids"]
            if input_ids.shape[1] < 10:
                continue
            
            try:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
                count += 1
            except Exception as e:
                print(f"  Error at sample {i}: {e}")
                continue
    
    if total_tokens == 0:
        return measure_ppl_synthetic(model)
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, total_loss, total_tokens


def measure_ppl_synthetic(model):
    """Fallback: measure PPL on synthetic data."""
    print("  Using synthetic data for PPL estimate")
    batch_size = 1
    seqlen = 512
    
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, model.config.vocab_size if hasattr(model, 'config') else 50277,
                                  (batch_size, seqlen))
        try:
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            loss = F.cross_entropy(
                logits[0, :-1].float(),
                input_ids[0, 1:],
                reduction='mean'
            )
            ppl = math.exp(loss.item())
            return ppl, loss.item() * (seqlen - 1), seqlen - 1
        except Exception as e:
            print(f"  Synthetic data error: {e}")
            return None, 0, 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-130m")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print(f"Loading Mamba2 model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        tokenizer = None
        print("  (no tokenizer available)")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    reference_mode = os.environ.get("REFERENCE", "0") == "1"
    
    print(f"\n{'='*60}")
    print(f"Measuring PPL with CHUNKED scan (reference)...")
    ppl_chunked = measure_ppl(model, tokenizer, max_samples=args.samples)
    if ppl_chunked[0] is not None:
        print(f"  PPL (chunked): {ppl_chunked[0]:.4f}")
    else:
        print(f"  PPL (chunked): FAILED")
        return
    
    print(f"\n{'='*60}")
    print(f"Patching to BLELLOCH scan...")
    original_fn = monkey_patch_mamba2(model)
    
    print(f"Measuring PPL with BLELLOCH scan...")
    ppl_blelloch = measure_ppl(model, tokenizer, max_samples=args.samples)
    
    print(f"\n{'='*60}")
    if ppl_blelloch[0] is not None:
        print(f"  PPL (chunked):  {ppl_chunked[0]:.4f}")
        print(f"  PPL (Blelloch): {ppl_blelloch[0]:.4f}")
        ppl_diff = ppl_blelloch[0] - ppl_chunked[0]
        print(f"  ΔPPL:           {ppl_diff:+.6f}")
        if abs(ppl_diff) < 0.01:
            print(f"  ✓ Blelloch matches chunked to within 0.01 PPL")
        elif ppl_blelloch[0] < ppl_chunked[0]:
            print(f"  ✗ Blelloch is LOWER (chunked overestimates PPL)")
        else:
            print(f"  ✗ Blelloch is HIGHER than chunked")
    
    print(f"\n{'='*60}")
    from selective_scan_blelloch import test_blelloch
    print(f"Running numerical verification...")
    test_blelloch()
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
