def merge_transcripts(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        current_line = ""
        
        for line in f_in:
            # 去除行首尾的空白字符和换行符
            line = line.strip()
            if not line:
                continue  # 跳过空行
                
            # 如果当前行包含 'mp3.pt_'，说明是一个新的音频对应的文本
            if "mp3.pt_" in line:
                # 如果缓冲区里已经有内容了，就把它写入文件并换行
                if current_line:
                    f_out.write(current_line + "\n")
                # 重新开始记录新的一行
                current_line = line
            else:
                # 如果不包含，说明是上一行的延续，直接追加到末尾
                # 如果句子之间需要空格，可以换成 current_line += " " + line
                if current_line:
                    current_line += line
                else:
                    current_line = line  # 处理文件开头第一行就没有 mp3.pt_ 的异常情况
                    
        # 循环结束后，别忘了把最后留在缓冲区里的一行写进去
        if current_line:
            f_out.write(current_line + "\n")

# 使用示例：
# 假设你的原文本叫 input.txt，处理后的文本叫 output.txt
input_file = r"D:\CODE\F5_like_TTS\fgo_proceed\content.txt"
output_file = r"D:\CODE\F5_like_TTS\fgo_proceed\content_fixed.txt"

merge_transcripts(input_file, output_file)
print("文本合并完成！")