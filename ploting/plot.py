import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# French      : 95.50% (191/200)
# Polish      : 99.50% (199/200)
# Portuguese  : 97.50% (195/200)
# Italian     : 94.50% (189/200)
# Spanish     : 95.50% (191/200)

data = {
    'Language': ['French', 'Polish', 'Portuguese', 'Italian', 'Spanish'],
    'Correct': [191, 199, 195, 189, 191],
    'Incorrect': [9, 1, 5, 11, 9]
}

def plot_prediction_results(data):
    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    x = range(len(df))
    width = 0.6

    bars1 = plt.bar(x, df['Correct'], width, label='Correct', color='#2ecc71')
    bars2 = plt.bar(x, df['Incorrect'], width, bottom=df['Correct'], label='Incorrect', color='#e74c3c')
    setPlot(df,x)
    plt.show()


def setPlot(df,X):
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Prediction Results by Language', fontsize=16)
    plt.xticks(X, df['Language'])
    plt.ylim(0, 210)
    plt.legend(fontsize=12)

    add_value_labels(df)
    plt.tight_layout()

def add_value_labels(df):
    for i, (correct, incorrect) in enumerate(zip(df['Correct'], df['Incorrect'])):
        # Correct count at bottom
        plt.text(i, correct/2, str(correct), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        # Incorrect count in the top section
        plt.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center', fontsize=11, fontweight='bold', color='white')


plot_prediction_results(data)