from pathlib import Path

data_dir = Path("data")
part1 = data_dir / "train_part1_4k.bin"
part2 = data_dir / "train_part2_4k.bin"
output = data_dir / "train_4k.bin"

with open(output, "wb") as out_f:
    for part in [part1, part2]:
        with open(part, "rb") as in_f:
            out_f.write(in_f.read())

# Verify: output size should equal the sum of the two input sizes
size1 = part1.stat().st_size
size2 = part2.stat().st_size
size_out = output.stat().st_size

print(f"{part1.name}: {size1:,} bytes")
print(f"{part2.name}: {size2:,} bytes")
print(f"Sum: {size1 + size2:,} bytes")
print(f"{output.name}: {size_out:,} bytes")
print("MATCH!" if size_out == size1 + size2 else "MISMATCH — something went wrong!")