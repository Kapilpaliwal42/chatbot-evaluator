from tools import list_files, read_file, write_file, run_command


print("Projects:")
print(list_files("."))


print("\nFiles in test-project:")
print(list_files("test-project"))


print("\nContents:")
print(read_file("test-project/hello.py"))

print("\nTrying to escape workspace...")

try:
    print(read_file("../../ollama_client.py"))
except Exception as e:
    print(type(e).__name__, ":", e)

write_file(
    "test-project/generated/example.txt",
    "Created inside a new directory."
)

print(read_file("test-project/hello.py"))

print("\nRunning hello.py...")

result = run_command(
    "python test-project/hello.py"
)

print(result)