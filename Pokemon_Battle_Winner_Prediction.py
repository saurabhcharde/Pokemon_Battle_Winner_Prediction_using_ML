
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


pokemon=pd.read_csv('C:/Users/sony/Desktop/AI Enigma/pokemons_data.csv',index_col=0)
combats=pd.read_csv('C:/Users/sony/Desktop/AI Enigma/combats.csv')


# In[7]:


pokemon.head()


# In[10]:


cols = ["First_pokemon","Second_pokemon","Winner"]
new_combat_data=combats[cols].replace(pokemon.Name)
new_combat_data.head()


# In[11]:


combats.Winner[combats.Winner == combats.First_pokemon] = 0
combats.Winner[combats.Winner == combats.Second_pokemon] = 1


# In[14]:


def normalization(data_df):
    stats=["Hit Points","Attack","Defense","Sp. Atk","Sp. Def","Speed","Legendary"]
    stats_df=pokemon[stats].T.to_dict("list")
    one=data_df.First_pokemon.map(stats_df)
    two=data_df.Second_pokemon.map(stats_df)
    temp_list=[]
    for i in range(len(one)):
        temp_list.append(np.array(one[i])-np.array(two[i]))
    new_test = pd.DataFrame(temp_list, columns=stats)
    for c in stats:
        description=new_test[c].describe()
        new_test[c]=(new_test[c]-description['min'])/(description['max']-description['min'])
    return new_test


# In[15]:


data=normalization(combats)
data = pd.concat([data,combats.Winner], axis=1)


# In[16]:


x_label=data.drop("Winner",axis=1)
y_label=data["Winner"]


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size=0.25, random_state=42)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(x_train, y_train)
pred = model.predict(x_test)
print('Accuracy of ', accuracy_score(pred, y_test)*100)


# In[19]:


test_data=pd.read_csv('C:/Users/sony/Desktop/AI Enigma/tests.csv')


# In[20]:


new_test_data=test_data[["First_pokemon","Second_pokemon"]].replace(pokemon.Name)
new_test_data.head()


# In[21]:


final_data=normalization(test_data)
pred=model.predict(final_data)
test_data["Winner"]=[test_data["First_pokemon"][i] if pred[i]==0 else test_data["Second_pokemon"][i] for i in range(len(pred))]


# In[24]:


combats_name = test_data[cols].replace(pokemon.Name)
combats_name[63:64]

