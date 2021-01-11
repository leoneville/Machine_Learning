
# coding: utf-8

# In[1]:

#Importando as bibliotecas
import pandas as pd


# In[18]:

def read_files(path, name_file, year_date, type_file):
    _file = f'{path}{name_file}{year_date}.{type_file}'
    colspecs = [(2, 10),
                (10, 12),
                (12, 24),
                (27, 39),
                (56, 69),
                (69, 82),
                (82, 95),
                (108, 121),
                (152, 170),
                (170, 188)


    ]

    names = ['data_pregao', 'codbdi', 'sigla_acao', 'nome_acao', 'preco_abertura', 'preco_maximo', 'preco_minimo', 'preco_fechamento', 'qnt_negocios', 'volume_negocios']

    df = pd.read_fwf(_file, colspecs = colspecs, names = names, skiprows = 1)
    return df


# In[13]:

#Filtrar ações
def filter_stocks(df):
    df = df [df['codbdi']==2]
    df = df.drop(['codbdi'], 1)
    return df


# In[14]:

#Ajuste campo de data
def parse_date(df):
    df['data_pregao'] = pd.to_datetime(df['data_pregao'], format = '%Y%m%d')
    return df


# In[15]:

#Ajuste dos campos numéricos
def parse_values(df):
    df['preco_abertura'] = (df['preco_abertura'] /100).astype(float)
    df['preco_minimo'] = (df['preco_minimo'] /100).astype(float)
    df['preco_maximo'] = (df['preco_maximo'] /100).astype(float)
    df['preco_fechamento'] = (df['preco_fechamento'] /100).astype(float)
    df['qnt_negocios'] = df['qnt_negocios'].astype(int)
    df['volume_negocios'] = df['volume_negocios'].astype(int)
    
    return df


# In[9]:

df.dtypes


# In[16]:

#Juntando os arquivos

def concat_files(path, name_file, year_date, type_file, final_file):
    for i, y in enumerate(year_date):
        df = read_files(path, name_file, y, type_file)
        df = filter_stocks(df)
        df = parse_date(df)
        df = parse_values(df)
        
        if i == 0:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])
            
    df_final.to_csv(f'{path}//{final_file}', index=False)
    


# In[22]:

#Executando programa de etl
year_date=[ '2018', '2019', '2020' ]

path=f'/home/leonardo/Artificial Intelligence/Machine Learning/Data Engineer/'

name_file='COTAHIST_A'

type_file='TXT'

final_file='all_bovespa.csv'

concat_files( path, name_file, year_date, type_file, final_file )


# In[ ]:



