import tkinter as tk
from tkinter import ttk


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
        for F in (Parameters, FunctionFlowGraph, Page2):
            frame = F(container, self)
            # initializing frame of that object from
            # Parameters, FunctionFlowGraph, page2 respectively with
            # for loop
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Parameters)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# first window frame Parameters
class Parameters(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Changing main frames buttons
        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Function Flow Chart",
                                 command=lambda: controller.show_frame(FunctionFlowGraph))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Page 2",
                             command=lambda: controller.show_frame(Page2))
        button2.grid(row=0, column=2, padx=5, pady=5)

        # Main parameters frame
        label_frame = ttk.LabelFrame(
            self, text="Population parameters")
        label_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=4)

        # Population size row
        pop_size_str = ttk.Label(label_frame, text='Population size')
        pop_size_str.grid(row=0, column=0, padx=5, pady=3)

        population_size = tk.IntVar()
        population_size.trace("w", lambda name, index,
                              mode, sv=population_size: self.callback(sv))
        pop_entry = ttk.Entry(
            label_frame, textvariable=population_size, justify='right')
        pop_entry.grid(row=0, column=1, padx=5, pady=3)

        # Parent percentage row
        parent_str = ttk.Label(label_frame, text='Parents [%]')
        parent_str.grid(row=1, column=0, padx=5, pady=3)

        parent_percentage = tk.IntVar()
        parent_percentage.trace("w", lambda name, index,
                                mode, sv=parent_percentage: self.callback(sv))
        par_entry = ttk.Entry(
            label_frame, textvariable=parent_percentage, justify='right')
        par_entry.grid(row=1, column=1, padx=5, pady=3)

        # Mutation percentage row
        mutation_str = ttk.Label(label_frame, text='Mutation chance [%]')
        mutation_str.grid(row=2, column=0, padx=5, pady=3)

        mutation_percentage = tk.IntVar()
        mutation_percentage.trace("w", lambda name, index,
                                  mode, sv=mutation_percentage: self.callback(sv))
        mut_entry = ttk.Entry(
            label_frame, textvariable=mutation_percentage, justify='right')
        mut_entry.grid(row=2, column=1, padx=5, pady=3)

        # Number of generations row
        generations_str = ttk.Label(label_frame, text='Number of generations')
        generations_str.grid(row=3, column=0, padx=5, pady=3)

        number_of_generations = tk.IntVar()
        number_of_generations.trace("w", lambda name, index,
                                    mode, sv=number_of_generations: self.callback(sv))
        num_of_gen_entry = ttk.Entry(
            label_frame, textvariable=number_of_generations, justify='right')
        num_of_gen_entry.grid(row=3, column=1, padx=5, pady=3)

        # Mutation and Crossover frame
        mutation_frame = ttk.LabelFrame(
            label_frame, text="Mutation", width=100, height=50)
        mutation_frame.grid(row=4, column=0, padx=5, pady=5)

        crossover_frame = ttk.LabelFrame(
            label_frame, text="Crossover", width=100, height=50)
        crossover_frame.grid(row=4, column=1, padx=5, pady=5)

        # Mutation frame (swap, inversion, scramble)
        swap_mutation = tk.IntVar()
        swap_mutation.trace("w", lambda name, index,
                            mode, sv=swap_mutation: self.callback(sv))
        swap_mutation_button = ttk.Checkbutton(mutation_frame, text='Swap',
                                               variable=swap_mutation)
        swap_mutation_button.grid(row=0, column=0, padx=5, pady=2, sticky='w')

        inverse_mutation = tk.IntVar()
        inverse_mutation.trace("w", lambda name, index,
                               mode, sv=inverse_mutation: self.callback(sv))
        inverse_mutation_button = ttk.Checkbutton(mutation_frame, text='Inverse',
                                                  variable=inverse_mutation)
        inverse_mutation_button.grid(
            row=1, column=0, padx=5, pady=2, sticky='w')

        scramble_mutation = tk.IntVar()
        scramble_mutation.trace("w", lambda name, index,
                                mode, sv=scramble_mutation: self.callback(sv))
        scramble_mutation_button = ttk.Checkbutton(mutation_frame, text='Scramble',
                                                   variable=scramble_mutation)
        scramble_mutation_button.grid(
            row=3, column=0, padx=5, pady=2, sticky='w')

        # Crossover (PMX, OX, CX)
        PMX_crossover = tk.IntVar()
        PMX_crossover.trace("w", lambda name, index,
                            mode, sv=PMX_crossover: self.callback(sv))
        PMX_crossover_button = ttk.Checkbutton(crossover_frame, text='PMX',
                                               variable=PMX_crossover)
        PMX_crossover_button.grid(row=0, column=0, padx=5, pady=2, sticky='w')

        OX_crossover = tk.IntVar()
        OX_crossover.trace("w", lambda name, index,
                           mode, sv=OX_crossover: self.callback(sv))
        OX_crossover_button = ttk.Checkbutton(crossover_frame, text='OX',
                                              variable=OX_crossover)
        OX_crossover_button.grid(row=1, column=0, padx=5, pady=2, sticky='w')

        CX_crossover = tk.IntVar()
        CX_crossover.trace("w", lambda name, index,
                           mode, sv=CX_crossover: self.callback(sv))
        CX_crossover_button = ttk.Checkbutton(crossover_frame, text='CX',
                                              variable=CX_crossover)
        CX_crossover_button.grid(row=3, column=0, padx=5, pady=2, sticky='w')

        # Selection Frame and buttons (Roulette, Ranking)
        selection_frame = ttk.LabelFrame(
            label_frame, text="Selection", width=100, height=50)
        selection_frame.grid(row=5, column=0, padx=5, pady=5, columnspan=2)

        selection = tk.StringVar()
        selection.trace("w", lambda name, index,
                        mode, sv=selection: self.callback(sv))
        roulette_selection_radiobutton = ttk.Radiobutton(
            selection_frame, text='Roulette', variable=selection, value='roulette')
        roulette_selection_radiobutton.grid(row=0, column=0, padx=5, pady=2)

        ranking_selection_radiobutton = ttk.Radiobutton(
            selection_frame, text='Ranking', variable=selection, value='ranking')
        ranking_selection_radiobutton.grid(row=0, column=1, padx=5, pady=2)

    def callback(self, sv):
        """Return value of entered value"""
        return print(sv._name, sv.get())


# second window frame FunctionFlowGraph
class FunctionFlowGraph(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Function Flow Chart",
                                 command=lambda: controller.show_frame(FunctionFlowGraph))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Page 2",
                             command=lambda: controller.show_frame(Page2))
        button2.grid(row=0, column=2, padx=5, pady=5)

        # label of frame Layout 2
        label = ttk.Label(self, text="Function Flow Chart")
        label.grid(row=1, column=1, padx=5, pady=5)


# third window frame page2
class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = ttk.Label(self, text="Page2")
        label.grid(row=1, column=1, padx=5, pady=5)

        param_frame = ttk.Button(self, text="Parameters",
                                 command=lambda: controller.show_frame(Parameters))
        param_frame.grid(row=0, column=0, padx=5, pady=5)

        chart_frame = ttk.Button(self, text="Function Flow Chart",
                                 command=lambda: controller.show_frame(FunctionFlowGraph))
        chart_frame.grid(row=0, column=1, padx=5, pady=5)

        button2 = ttk.Button(self, text="Page 2",
                             command=lambda: controller.show_frame(Page2))
        button2.grid(row=0, column=2, padx=5, pady=5)


# Driver Code
app = tkinterApp()
app.mainloop()
