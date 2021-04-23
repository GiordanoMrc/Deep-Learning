import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



truePositive = 50
trueNegative = 20
falsePositive = 50
falseNegative = 20

array = [[truePositive,falseNegative],[falsePositive,trueNegative]]
df_cm = pd.DataFrame(array, index = [i for i in ["Maligna","Benigna"]],
                  columns = ["Maligna","Benigna"])

plt.figure(figsize = (10,7))
plt.title("Matriz de Confusão", fontsize=25)
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,cmap="Blues")
plt.xlabel('Classe Predita',fontsize=20)
plt.ylabel('Classe Real',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18, labelbottom = True, bottom=False, top = False, labeltop=False)
plt.show()