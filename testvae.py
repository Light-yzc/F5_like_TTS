import pandas as pd
df = pd.read_parquet(r'D:\CODE\table.parquet')
print(df.head()) # 这里可以看到每个音频对应的角色名、文本等信息