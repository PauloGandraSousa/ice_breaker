from unstructured.partition.auto import partition

elements = partition(filename="docs/examples/simple.txt")
print("\n\n".join([str(el) for el in elements]))