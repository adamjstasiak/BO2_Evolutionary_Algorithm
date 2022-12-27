import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):

        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.title(self, "Genetic Algorithm")

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (Parameters, GraphicalSolution, FunctionFlowGraph):
            frame = F(container, self)
            # initializing frame of that object from
            # Parameters, GraphicalSolution, FunctionFlowGraph respectively with
            # for loop
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Parameters)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def get_page(self, page_class):
        return self.frames[page_class]


# first window frame Parameters
class Parameters(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Changing main frames buttons
        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Graphical Solution",
                                 command=lambda: controller.show_frame(GraphicalSolution))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Function Flow Graph",
                             command=lambda: controller.show_frame(FunctionFlowGraph))
        button2.grid(row=0, column=2, padx=5, pady=5)

        # Main parameters frame
        label_frame = ttk.LabelFrame(
            self, text="Population parameters")
        label_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=4)

        # Population size row
        pop_size_str = ttk.Label(label_frame, text='Population size')
        pop_size_str.grid(row=0, column=0, padx=5, pady=3)

        self.population_size = tk.IntVar()
        self.population_size.trace("w", lambda name, index,
                                   mode, sv=self.population_size: self.callback(sv))
        pop_entry = ttk.Entry(
            label_frame, textvariable=self.population_size, justify='right')
        pop_entry.grid(row=0, column=1, padx=5, pady=3)

        # Parent percentage row
        parent_str = ttk.Label(label_frame, text='Parents [%]')
        parent_str.grid(row=1, column=0, padx=5, pady=3)

        self.parent_percentage = tk.IntVar()
        self.parent_percentage.trace("w", lambda name, index,
                                     mode, sv=self.parent_percentage: self.callback(sv))
        par_entry = ttk.Entry(
            label_frame, textvariable=self.parent_percentage, justify='right')
        par_entry.grid(row=1, column=1, padx=5, pady=3)

        # Mutation percentage row
        mutation_str = ttk.Label(label_frame, text='Mutation chance [%]')
        mutation_str.grid(row=2, column=0, padx=5, pady=3)

        self.mutation_percentage = tk.IntVar()
        self.mutation_percentage.trace("w", lambda name, index,
                                       mode, sv=self.mutation_percentage: self.callback(sv))
        mut_entry = ttk.Entry(
            label_frame, textvariable=self.mutation_percentage, justify='right')
        mut_entry.grid(row=2, column=1, padx=5, pady=3)

        # Number of generations row
        generations_str = ttk.Label(label_frame, text='Number of generations')
        generations_str.grid(row=3, column=0, padx=5, pady=3)

        self.number_of_generations = tk.IntVar()
        self.number_of_generations.trace("w", lambda name, index,
                                         mode, sv=self.number_of_generations: self.callback(sv))
        num_of_gen_entry = ttk.Entry(
            label_frame, textvariable=self.number_of_generations, justify='right')
        num_of_gen_entry.grid(row=3, column=1, padx=5, pady=3)

        # Mutation and Crossover frame
        mutation_frame = ttk.LabelFrame(
            label_frame, text="Mutation", width=100, height=50)
        mutation_frame.grid(row=4, column=0, padx=5, pady=5)

        crossover_frame = ttk.LabelFrame(
            label_frame, text="Crossover", width=100, height=50)
        crossover_frame.grid(row=4, column=1, padx=5, pady=5)

        # Mutation frame (swap, inversion, scramble)
        self.swap_mutation = tk.IntVar()
        self.swap_mutation.trace("w", lambda name, index,
                                 mode, sv=self.swap_mutation: self.callback(sv))
        swap_mutation_button = ttk.Checkbutton(mutation_frame, text='Swap',
                                               variable=self.swap_mutation)
        swap_mutation_button.grid(row=0, column=0, padx=5, pady=2, sticky='w')

        self.inverse_mutation = tk.IntVar()
        self.inverse_mutation.trace("w", lambda name, index,
                                    mode, sv=self.inverse_mutation: self.callback(sv))
        inverse_mutation_button = ttk.Checkbutton(mutation_frame, text='Inverse',
                                                  variable=self.inverse_mutation)
        inverse_mutation_button.grid(
            row=1, column=0, padx=5, pady=2, sticky='w')

        self.scramble_mutation = tk.IntVar()
        self.scramble_mutation.trace("w", lambda name, index,
                                     mode, sv=self.scramble_mutation: self.callback(sv))
        scramble_mutation_button = ttk.Checkbutton(mutation_frame, text='Scramble',
                                                   variable=self.scramble_mutation)
        scramble_mutation_button.grid(
            row=3, column=0, padx=5, pady=2, sticky='w')

        # Crossover (PMX, OX, CX)
        self.PMX_crossover = tk.IntVar()
        self.PMX_crossover.trace("w", lambda name, index,
                                 mode, sv=self.PMX_crossover: self.callback(sv))
        PMX_crossover_button = ttk.Checkbutton(crossover_frame, text='PMX',
                                               variable=self.PMX_crossover)
        PMX_crossover_button.grid(row=0, column=0, padx=5, pady=2, sticky='w')

        self.OX_crossover = tk.IntVar()
        self.OX_crossover.trace("w", lambda name, index,
                                mode, sv=self.OX_crossover: self.callback(sv))
        OX_crossover_button = ttk.Checkbutton(crossover_frame, text='OX',
                                              variable=self.OX_crossover)
        OX_crossover_button.grid(row=1, column=0, padx=5, pady=2, sticky='w')

        self.CX_crossover = tk.IntVar()
        self.CX_crossover.trace("w", lambda name, index,
                                mode, sv=self.CX_crossover: self.callback(sv))
        CX_crossover_button = ttk.Checkbutton(crossover_frame, text='CX',
                                              variable=self.CX_crossover)
        CX_crossover_button.grid(row=3, column=0, padx=5, pady=2, sticky='w')

        # Selection Frame and buttons (Roulette, Ranking)
        selection_frame = ttk.LabelFrame(
            label_frame, text="Selection", width=100, height=50)
        selection_frame.grid(row=5, column=0, padx=5, pady=5, columnspan=2)

        self.selection = tk.StringVar()
        self.selection.trace("w", lambda name, index,
                             mode, sv=self.selection: self.callback(sv))
        roulette_selection_radiobutton = ttk.Radiobutton(
            selection_frame, text='Roulette', variable=self.selection, value='roulette')
        roulette_selection_radiobutton.grid(row=0, column=0, padx=5, pady=2)

        ranking_selection_radiobutton = ttk.Radiobutton(
            selection_frame, text='Ranking', variable=self.selection, value='ranking')
        ranking_selection_radiobutton.grid(row=0, column=1, padx=5, pady=2)

        # Start
        s = ttk.Style()
        s.configure('my.TButton', font=('Helvetica', 20))
        self.start = ttk.Button(label_frame, text="Start", style='my.TButton',
                                command=self.start_algorithm)
        self.start.grid(row=6, column=0, padx=5, pady=5, columnspan=2)

        # Write solution
        self.l = tk.Listbox(self, height=5, width=45)
        self.l.grid(column=0, row=3, columnspan=3)

        # Canvas
        self.canv = tk.Canvas(self, width=650, height=400, background='pink')
        self.canv.grid(row=1, column=4, padx=5,
                       pady=5, rowspan=2, columnspan=2)
        self.canv.bind("<ButtonPress-1>", self.paint_parcels)

        pop_size_str = ttk.Label(self, text='Number of parcels to choose')
        pop_size_str.grid(row=0, column=4, padx=5, pady=3, sticky='E')

        # additional, not necessary - number of parcels can be bigger than fabrics
        self.number_of_parcels = tk.IntVar()
        self.number_of_parcels.trace("w", lambda name, index,
                                     mode, sv=self.number_of_parcels: self.callback(sv))
        parcels_entry = ttk.Entry(
            self, textvariable=self.number_of_parcels, justify='right')
        parcels_entry.grid(row=0, column=5, padx=5, pady=3, sticky='W')

        # Reset
        self.reset = ttk.Button(self, text="reset", style='my.TButton',
                                command=self.reset_canvas)
        self.reset.grid(row=2, column=0, padx=5, pady=5, columnspan=3)

        self.parcel_distances = []
        self.increment = 0

    def distance_matrix_from_points(self):
        matrix = np.matrix(
            np.ones((len(self.parcel_distances), len(self.parcel_distances))))
        for i, el_1 in enumerate(self.parcel_distances):
            for j, el_2 in enumerate(self.parcel_distances):
                matrix[i, j] = np.sqrt(
                    (el_1[0] - el_2[0])**2 + (el_1[1] - el_2[1])**2)
        np.fill_diagonal(matrix, np.inf)
        return matrix

    def reset_canvas(self):
        self.canv.delete('all')
        self.parcel_distances.clear()
        self.l.delete(0, tk.END)
        self.increment = 0

    def paint_parcels(self, event):
        self.parcel_distances.append([event.x, event.y])
        x1, y1 = (event.x - 12), (event.y - 12)
        x2, y2 = (event.x + 12), (event.y + 12)
        self.canv.create_oval(x1, y1, x2, y2, fill='green')
        self.canv.create_text(event.x, event.y, text=str(
            self.increment), fill="black", font=('Helvetica 15 bold'))
        self.increment += 1

    def paint_factories(self):
        for i, el in enumerate(self.parcel_distances):
            self.canv.create_text(el[0]+15, el[1]-15, text=str(
                i), fill="magenta", font=('Helvetica 15 bold'))

    def callback(self, sv):
        """Return value of entered value"""
        return print(sv._name, sv.get())

    def start_algorithm(self):
        graph_page = self.controller.get_page(FunctionFlowGraph)
        graph_page.plot_dataframe(graph_page.canvas, graph_page.ax, df)
        if self.l.size() > 4:
            self.l.delete(0)
        if len(self.parcel_distances) > 1:
            self.l.insert('end', 'Solution: %d' % 1)
        else:
            self.l.insert('end', 'Input at least two points')
        print(self.parcel_distances)
        distance_matrix = self.distance_matrix_from_points()
        self.paint_factories()
        print(distance_matrix)


# second window frame GraphicalSolution
class GraphicalSolution(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Graphical Solution",
                                 command=lambda: controller.show_frame(GraphicalSolution))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Function Flow Graph",
                             command=lambda: controller.show_frame(FunctionFlowGraph))
        button2.grid(row=0, column=2, padx=5, pady=5)

        # Canvas
        self.canv_solution = tk.Canvas(
            self, width=650, height=400, background='pink')
        self.canv_solution.grid(row=1, column=0, padx=5,
                                pady=5, rowspan=2, columnspan=2000)
        # self.canv.bind("<ButtonPress-1>", self.paint_parcels)


# third window frame FunctionFlowGraph
class FunctionFlowGraph(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Graphical Solution",
                                 command=lambda: controller.show_frame(GraphicalSolution))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Function Flow Graph",
                             command=lambda: controller.show_frame(FunctionFlowGraph))
        button2.grid(row=0, column=2, padx=5, pady=5)

        s = ttk.Style()
        s.configure('My.TFrame', background='green')
        self.plot_frame = ttk.Frame(
            self, style='My.TFrame', width=20, height=20)
        self.plot_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2000)

        # Space for graph
        figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        self.canvas = FigureCanvasTkAgg(figure, self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        self.canvas.draw()

    def plot_dataframe(self, canvas, ax, df):
        ax.clear()         # clear axes from previous plot
        ax.plot(df.year, df.unemployment_rate)
        canvas.draw()


df = {'year': [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
      'unemployment_rate': [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
      }
df = pd.DataFrame(df)


def main():

    app = tkinterApp()
    # print('bbbbb', app.get_page(Parameters).population_size.get())
    app.mainloop()


if __name__ == "__main__":
    main()
