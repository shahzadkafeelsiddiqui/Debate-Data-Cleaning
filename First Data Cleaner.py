
import pandas as pd
df = pd.DataFrame()
with open('US2016G1tv.txt', errors='ignore') as file:
    for item in file:
        print (item)
        if "HOLT:" in item: 
            value = item.split(":")
            person = value [0]
            text = value [1]
            df = df.append({'Person': value[0], 'Text':value[1]}, ignore_index = True)
            
        elif "CLINTON:" in item: 
            
            value = item.split(":")
            person = value [0]
            text = value [1]
            df = df.append({'Person': value[0], 'Text': value[1]}, ignore_index = True)
        elif "TRUMP:" in item: 
            
            value = item.split(":")
            person = value [0]
            text = value [1]
            df = df.append({'Person': value[0], 'Text': value[1]}, ignore_index = True)
        else:
            
            df = df.append({'Person': df['Person'].iloc[-1], 'Text': item}, ignore_index = True)
                    
    print (df)  
    df.to_csv('Data.csv', index="False")
