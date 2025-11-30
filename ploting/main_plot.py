from . import plot

data = {
    'Language': ['French', 'Polish', 'Portuguese', 'Italian', 'Spanish'],
    'Correct': [191, 199, 195, 189, 191],
    'Incorrect': [9, 1, 5, 11, 9]
}

def main():
    plotter = plot.Plotter(data)
    plotter.plot_prediction_results()

if __name__ == '__main__':
    main()
