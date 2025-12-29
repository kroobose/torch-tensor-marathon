
import torch

def debug_indexing():
    try:
        x = torch.arange(10)
        print(f"x type: {type(x)}")
        print(f"x shape: {x.shape}")

        # Try negative step slicing
        result = x[::-1]
        print("x[::-1] success")
        print(f"Result: {result}")

        # Try explicit slice
        result2 = x[slice(None, None, -1)]
        print("Slice success")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_indexing()
