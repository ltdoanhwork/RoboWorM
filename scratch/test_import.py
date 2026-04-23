try:
    from methods.cache_strategy.FasterCache.runtime import apply_fastercache
    print("SUCCESS: apply_fastercache found")
except ImportError as e:
    print(f"FAILURE: {e}")
