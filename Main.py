with open("chat.txt", "r", encoding="utf-8") as f:
    content = f.read().splitlines()

# 1. Initialize empty dictionary
out = {}

# 2. Iterate through lines using 'line' (not 'i', to avoid confusion)
for line in content:
    k = 0
    
    # 3. Find the colon (using 'idx' for the number)
    # We start at 16 to skip the timestamp part
    for idx in range(16, len(line)):
        if line[idx] == ":":
            k = idx
            break
    
    # 4. SAFETY CHECK: Only proceed if a colon was actually found
    # (This skips system messages like "You were added")
    if k != 0:
        name = line[17:k]       # Extract Name
        message = line[k+2:]    # Extract Message (+2 skips the ": " part)

        # 5. Add to Dictionary
        if name in out:
            # Append new message with a space separator
            out[name] += " " + message 
        else:
            # Create new entry
            out[name] = message

print(out)