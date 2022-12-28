import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from genetic_algorithm import genetic_algorithm
import function as fun


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
        for F in (Parameters, FunctionFlowGraph):
            frame = F(container, self)
            # initializing frame of that object from
            # Parameters, FunctionFlowGraph respectively with
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

        graph_frame = ttk.Button(self, text="Function Flow Graph",
                                 command=lambda: controller.show_frame(FunctionFlowGraph))
        graph_frame.grid(row=0, column=1, padx=5, pady=5)

        # Main parameters frame
        label_frame = ttk.LabelFrame(
            self, text="Population parameters")
        label_frame.grid(row=1, column=0, padx=20, pady=5, columnspan=4)

        # Population size row
        pop_size_str = ttk.Label(label_frame, text='Population size')
        pop_size_str.grid(row=0, column=0, padx=5, pady=3)

        self.population_size = tk.IntVar()
        self.population_size.trace("w", lambda name, index,
                                   mode, sv=self.population_size: self.callback(sv))
        self.population_size.set(20)
        pop_entry = ttk.Entry(
            label_frame, textvariable=self.population_size, justify='right')
        pop_entry.grid(row=0, column=1, padx=5, pady=3)

        # Parent percentage row
        parent_str = ttk.Label(label_frame, text='Selection size [%]')
        parent_str.grid(row=1, column=0, padx=5, pady=3)

        self.selection_size = tk.IntVar()
        self.selection_size.trace("w", lambda name, index,
                                  mode, sv=self.selection_size: self.callback(sv))
        self.selection_size.set(50)
        par_entry = ttk.Entry(
            label_frame, textvariable=self.selection_size, justify='right')
        par_entry.grid(row=1, column=1, padx=5, pady=3)

        # Mutation percentage row
        mutation_str = ttk.Label(label_frame, text='Mutation chance [%]')
        mutation_str.grid(row=2, column=0, padx=5, pady=3)

        self.mutation_percentage = tk.IntVar()
        self.mutation_percentage.trace("w", lambda name, index,
                                       mode, sv=self.mutation_percentage: self.callback(sv))
        self.mutation_percentage.set(5)
        mut_entry = ttk.Entry(
            label_frame, textvariable=self.mutation_percentage, justify='right')
        mut_entry.grid(row=2, column=1, padx=5, pady=3)

        # Number of generations row
        generations_str = ttk.Label(label_frame, text='Number of generations')
        generations_str.grid(row=3, column=0, padx=5, pady=3)

        self.number_of_generations = tk.IntVar()
        self.number_of_generations.trace("w", lambda name, index,
                                         mode, sv=self.number_of_generations: self.callback(sv))
        self.number_of_generations.set(100)
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
        self.swap_mutation.set(1)
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
        self.PMX_crossover.set(1)
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
        self.selection.set('roulette')
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

        # Canvas
        self.canv = tk.Canvas(self, width=745, height=520,
                              background='pink', borderwidth=10)
        self.canv.grid(row=1, column=4, padx=5,
                       pady=5, rowspan=3, columnspan=2)
        self.canv.bind("<ButtonPress-1>", self.paint_parcels)

        # Parcels frame
        parcel_frame = ttk.LabelFrame(
            self, text="Parcels")
        parcel_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=4)

        # additional, not necessary - number of parcels can be bigger than fabrics
        pop_size_str = ttk.Label(parcel_frame, text='Max number of parcels')
        pop_size_str.grid(row=0, column=0, padx=5,
                          pady=3, sticky='E')

        self.number_of_parcels = tk.IntVar()
        self.number_of_parcels.trace("w", lambda name, index,
                                     mode, sv=self.number_of_parcels: self.callback(sv))
        self.number_of_parcels.set(6)
        parcels_entry = ttk.Entry(
            parcel_frame, textvariable=self.number_of_parcels, justify='right')
        parcels_entry.grid(row=0, column=1, padx=5, pady=3, sticky='W')

        # Reset
        self.reset = ttk.Button(parcel_frame, text="reset", style='my.TButton',
                                command=self.reset_canvas)
        self.reset.grid(row=1, column=0, padx=5, pady=10, columnspan=3)

        # Write solution
        self.solution = tk.Listbox(self, height=5, width=46)
        self.solution.grid(column=0, row=3, padx=20, pady=10, columnspan=3)

        self.parcel_distances = []
        self.increment = 0

    def distance_matrix_from_points(self):
        matrix = np.matrix(
            np.ones((len(self.parcel_distances), len(self.parcel_distances))))
        for i, el_1 in enumerate(self.parcel_distances):
            for j, el_2 in enumerate(self.parcel_distances):
                matrix[i, j] = int(np.sqrt(
                    (el_1[0] - el_2[0])**2 + (el_1[1] - el_2[1])**2))
        np.fill_diagonal(matrix, np.inf)
        return np.array(matrix)

    def reset_canvas(self):
        self.canv.delete('all')
        self.parcel_distances.clear()
        self.solution.delete(0, tk.END)
        self.increment = 0

    def paint_parcels(self, event):
        if self.increment < self.number_of_parcels.get():
            self.parcel_distances.append([event.x, event.y])
            x1, y1 = (event.x - 12), (event.y - 12)
            x2, y2 = (event.x + 12), (event.y + 12)
            self.canv.create_oval(x1, y1, x2, y2, fill='green')
            self.canv.create_text(event.x, event.y, text=str(
                self.increment), fill="black", font=('Helvetica 15 bold'))
            self.increment += 1

    # TODO: change to print on appropriate parcels
    def paint_factories(self):
        for i, el in enumerate(self.parcel_distances):
            self.canv.create_text(el[0]+15, el[1]-15, text=str(
                i), fill="magenta", font=('Helvetica 15 bold'))

    def callback(self, sv):
        """Return value of entered value"""
        return [sv._name, sv.get()]

    def start_algorithm(self):
        # graph_page
        graph_page = self.controller.get_page(FunctionFlowGraph)
        distance_matrix = self.distance_matrix_from_points()

        # TODO: Make factory list by size from other file (txt, csv, xls)
        factory_list = fun.create_fabric_list(6, 6)

        cross_probability = 1
        if self.PMX_crossover.get() == 1 or self.OX_crossover.get() == 1 or self.CX_crossover.get() == 1:
            cross_probability = 1
        else:
            cross_probability = 0
        mut_probability = 1
        if self.swap_mutation.get() == 1 or self.inverse_mutation.get() == 1 or self.scramble_mutation.get() == 1:
            mut_probability = 1
        else:
            mut_probability = 0

        if len(factory_list) > len(distance_matrix[0]):
            self.solution.insert(
                'end', 'Number of parcels is smaller than', 'number of fabrics', 'ADD NEW PARCELS')

        solution = genetic_algorithm(distance_matrix, flow_matrix, factory_list, self.population_size.get(),
                                     self.selection_size.get(), self.number_of_generations.get(), selection_type=self.selection.get(),
                                     crossover_probability=cross_probability, mutation_probability=mut_probability,
                                     pmx_probability=self.PMX_crossover.get(), cx_probability=self.CX_crossover.get(),
                                     ox_probability=self.OX_crossover.get(), swap_probability=self.swap_mutation.get(),
                                     inversion_probability=self.inverse_mutation.get(), scramble_probability=self.scramble_mutation.get())

        # TODO: Enter dataframe with min values from algorithm
        graph_page.plot_dataframe(graph_page.canvas, graph_page.ax, df)

        self.paint_factories()

        if self.solution.size() > 4:
            self.solution.delete(0)
        if len(self.parcel_distances) > 1:
            self.solution.insert('end', 'Solution: %s' % str(
                str(solution[0]) + '  sum: ' + str(solution[1])))
        else:
            self.solution.insert('end', 'Input at least two points')


# second window frame FunctionFlowGraph
class FunctionFlowGraph(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=22, pady=5)

        graph_frame = ttk.Button(self, text="Function Flow Graph",
                                 command=lambda: controller.show_frame(FunctionFlowGraph))
        graph_frame.grid(row=0, column=2, padx=21, pady=5, sticky='E')

        s = ttk.Style()
        s.configure('My.TFrame', background='green')
        self.plot_frame = ttk.Frame(
            self, style='My.TFrame', width=20, height=20)
        self.plot_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2000)

        # Space for graph
        figure = plt.Figure(figsize=(11, 6), dpi=100)
        self.ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        self.canvas = FigureCanvasTkAgg(figure, self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        self.canvas.draw()

    # TODO: change to get dataframe from other file
    def plot_dataframe(self, canvas, ax, df):
        ax.clear()         # clear axes from previous plot
        ax.plot(df.year, df.unemployment_rate)
        canvas.draw()


df = {'year': [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
      'unemployment_rate': [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
      }
df = pd.DataFrame(df)

flow = [[np.inf, 4, 2, 2, 3, 1],
        [4, np.inf, 3, 5, 5, 8],
        [2, 3, np.inf, 9, 6, 4],
        [2, 5, 9, np.inf, 7, 9],
        [3, 5, 6, 7, np.inf, 2],
        [1, 8, 4, 9, 2, np.inf]]
flow_matrix = np.array(flow)


def main():

    app = tkinterApp()
    # print('bbbbb', app.get_page(Parameters).population_size.get())
    app.mainloop()


if __name__ == "__main__":
    main()
