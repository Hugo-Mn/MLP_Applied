import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(self, data):
        self.data = data
        self.df =  df = pd.DataFrame(self.data)
        self.x = range(len(self.df))
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))


    def plot_prediction_results(self):
        width = 0.6

        bars1 = plt.bar(self.x, self.df['Correct'], width, label='Correct', color='#2ecc71')
        bars2 = plt.bar(self.x, self.df['Incorrect'], width, bottom=self.df['Correct'], label='Incorrect', color='#e74c3c')
        self.setPlot()
        plt.show()


    def setPlot(self):
        plt.xlabel('Language', fontsize=14)
        plt.ylabel('Number of Predictions', fontsize=14)
        plt.title('Prediction Results by Language', fontsize=16)
        plt.xticks(self.x, self.df['Language'])
        plt.ylim(0, 210)
        plt.legend(fontsize=12)

        self.add_value_labels()
        plt.tight_layout()

    def add_value_labels(self):
        for i, (correct, incorrect) in enumerate(zip(self.df['Correct'], self.df['Incorrect'])):
            # Correct count at bottom
            plt.text(i, correct/2, str(correct), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            # Incorrect count in the top section
            plt.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
