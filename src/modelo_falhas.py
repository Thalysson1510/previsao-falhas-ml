import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Gerar dados simulados
np.random.seed(42)
n = 500
data = {
    'temperatura': np.random.normal(70, 10, n),
    'pressao': np.random.normal(30, 5, n),
    'vibracao': np.random.normal(5, 1, n),
    'tempo_uso_horas': np.random.normal(1000, 200, n),
    'falha': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
}
df = pd.DataFrame(data)

# Treino e teste
X = df.drop('falha', axis=1)
y = df['falha']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Importância das features
importancias = modelo.feature_importances_
plt.barh(X.columns, importancias)
plt.xlabel('Importância')
plt.title('Importância das Variáveis')
plt.tight_layout()
plt.show()
