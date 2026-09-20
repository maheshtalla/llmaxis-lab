import os

os.makedirs("new_folder", exist_ok=True)

file_path = os.path.join("new_folder", "new_file.txt")

with open(file_path, "x+") as file:
    file.write("This is a new file created in the new_folder.")
    file.write("\nThis is the second line of the new file.")
    file.write("\nThis is the third line of the new file.")

    # Move the cursor to the beginning of the file to read its content
    file.seek(0)
    content = file.read()

print(content)