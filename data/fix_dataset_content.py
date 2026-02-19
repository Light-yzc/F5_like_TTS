import os

def main():
    content_file_path = 'content.txt'
    directory_path = 'jvs_proceed'

    # Check if files/directories exist
    if not os.path.exists(content_file_path):
        print(f"Error: {content_file_path} does not exist.")
        return
    if not os.path.exists(directory_path):
        print(f"Error: {directory_path} does not exist.")
        return

    # 1. Read content.txt and extract filenames
    content_filenames = set()
    try:
        with open(content_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('_')
                if len(parts) >= 2:
                    # According to instruction: "第一个下划线后面...是文件名"
                    # format: prefix_filename_transcript
                    filename = parts[1]
                    content_filenames.add(filename)
                else:
                    print(f"Warning: Line {line_num} in {content_file_path} has unexpected format: {line}")
    except Exception as e:
        print(f"Error reading {content_file_path}: {e}")
        return

    print(f"Found {len(content_filenames)} unique filenames in {content_file_path}")

    # 2. List files in jvs_proceed directory
    try:
        dir_filenames = set(os.listdir(directory_path))
    except Exception as e:
        print(f"Error listing directory {directory_path}: {e}")
        return

    print(f"Found {len(dir_filenames)} files in {directory_path}")

    # 3. Compare sets
    # Files in content.txt but NOT in directory (Missing files)
    missing_files = content_filenames - dir_filenames
    
    # Files in directory but NOT in content.txt (Extra files)
    extra_files = dir_filenames - content_filenames

    with open('dataset_check_result.txt', 'w', encoding='utf-8') as res_file:
        # 4. Report results
        msg = "\n" + "="*50 + "\n"
        msg += f"Files in {content_file_path} but missing from {directory_path} (Count: {len(missing_files)}):\n"
        msg += "="*50 + "\n"
        print(msg)
        res_file.write(msg)
        
        if missing_files:
            for f in sorted(list(missing_files)):
                print(f)
                res_file.write(f + "\n")
        else:
            msg = "None. All files in content.txt exist in the directory.\n"
            print(msg)
            res_file.write(msg)

        msg = "\n" + "="*50 + "\n"
        msg += f"Files in {directory_path} but not in {content_file_path} (Count: {len(extra_files)}):\n"
        msg += "="*50 + "\n"
        print(msg)
        res_file.write(msg)

        if extra_files:
            for f in sorted(list(extra_files)):
                print(f)
                res_file.write(f + "\n")
        else:
            msg = "None. No extra files in the directory.\n"
            print(msg)
            res_file.write(msg)
    
            msg = "None. No extra files in the directory.\n"
            print(msg)
            res_file.write(msg)
    
    print(f"Results saved to dataset_check_result.txt")

    # 5. Generate content_cleaned.txt
    valid_filenames = content_filenames - missing_files
    print(f"\nGenerating content_cleaned.txt with valid entries...")
    
    try:
        with open(content_file_path, 'r', encoding='utf-8') as f_in, \
             open('content_cleaned.txt', 'w', encoding='utf-8') as f_out:
            written_count = 0
            for line in f_in:
                line_content = line.strip()
                if not line_content:
                    continue
                
                parts = line_content.split('_')
                if len(parts) >= 2:
                    filename = parts[1]
                    if filename in valid_filenames:
                        f_out.write(line)
                        written_count += 1
        print(f"Successfully wrote {written_count} lines to content_cleaned.txt")
        
    except Exception as e:
        print(f"Error writing content_cleaned.txt: {e}")

if __name__ == "__main__":
    main()
