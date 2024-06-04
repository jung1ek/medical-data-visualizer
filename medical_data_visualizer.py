import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calculate_overweight(row):
    bmi = row['weight']/(row['height']/100)**2
    if bmi>25:
        return 1
    else:
        return 0

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df.apply(lambda row: calculate_overweight(row),axis=1)

# 3
df['cholesterol']= df['cholesterol'].apply(lambda x: 1 if x>1 else 0 )
df['gluc'] = df['gluc'].apply(lambda x: 1 if x>1 else 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,  id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


    # 6
    df_cat = df_cat
    

    # 7


    # 8
    fig, ax = plt.subplot()
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', height=5, aspect=1.2)
    fig.set_axis_labels("variable", "total")
    fig.set_titles("cardio = {col_name}")


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[df['ap_lo'] <= df['ap_hi']]
    filtered_df = df[df['height']>=df['height'].quantile(0.025)] # return the 2.5th percentile lowest value 
    filtered_df = filtered_df[filtered_df['height']<=filtered_df['height'].quantile(0.975)] # return the 97.5th percentile highest value
    filtered_df = filtered_df[filtered_df['weight']>=filtered_df['weight'].quantile(0.025)]
    filtered_df = filtered_df[filtered_df['weight']<=filtered_df['weight'].quantile(0.975)]

    # 12
    corr = filtered_df.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
