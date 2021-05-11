from __future__ import annotations
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd


class Context():
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def make_diagram(self, values: list, y: str, *args) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        plot = self._strategy.do_algorithm(values, y, *args)
        plt.show()


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, values: list, y: str, *args):
        pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class MakeLinePlot(Strategy):
    def do_algorithm(self, values: pd.DataFrame, y: str) -> None:
        plt.plot(range(len(values)), values)
        plt.title(f'Line Chart of {y} over time')
        plt.xlabel(f'Time in days starting from {values.index[0]} going to {values.index[len(values) - 1]}')
        plt.ylabel(y)
        plt.show()


class MakeScatterPlot(Strategy):
    def do_algorithm(self, values: pd.DataFrame, y: str) -> None:
        plt.scatter(range(len(values)), values)
        plt.title(f'Scatter Plot of {y} over time')
        plt.xlabel(f'Time in days starting from {values.index[0]} going to {values.index[len(values) - 1]}')
        plt.ylabel(y)
        plt.show()


class MakeBarDiagram(Strategy):
    def do_algorithm(self, values: pd.DataFrame, y: str) -> None:
        plt.bar(list(values.index), list(values), facecolor='blue', alpha=0.5)
        plt.xticks(rotation='vertical')
        plt.title(f'Bar diagram of {y}')
        plt.show()


class MakeHistogram(Strategy):
    def do_algorithm(self, values: pd.DataFrame, y: str) -> None:
        from mpl_toolkits.axes_grid1 import host_subplot
        import matplotlib.dates as md
        import mpl_toolkits.axisartist as AA
        data = values.copy()
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

        # prepare days and months axes
        fig = plt.figure(figsize=(16, 8))
        days = host_subplot(111, axes_class=AA.Axes, figure=fig)
        plt.subplots_adjust(bottom=0.1)
        months = days.twiny()

        # position months axis
        offset = -20
        new_fixed_axis = months.get_grid_helper().new_fixed_axis
        months.axis['bottom'] = new_fixed_axis(loc='bottom',
                                               axes=months,
                                               offset=(0, offset))
        months.axis['bottom'].toggle(all=True)

        # plot
        days.bar(data.index, data)

        # formatting days axis
        if (len(data) < 400):
            # formatting days axis
            days.xaxis.set_major_locator(md.DayLocator(interval=len(data) // 30))
            days.xaxis.set_major_formatter(md.DateFormatter('%d'))
            plt.setp(days.xaxis.get_majorticklabels(), rotation=0)
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting months axis
            months.xaxis.set_major_locator(md.MonthLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%b'))
            months.set_xlim([data.index[0], data.index[-1]])
        else:
            days.xaxis.set_major_locator(md.MonthLocator(interval=len(data) // (30 * 20)))
            days.xaxis.set_major_formatter(md.DateFormatter('%b'))
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting months axis
            months.xaxis.set_major_locator(md.YearLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            months.set_xlim([data.index[0], data.index[-1]])

        plt.show()


class MakeDetailedScatter(Strategy):
    def do_algorithm(self, values: pd.DataFrame, y: str) -> None:
        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA
        import matplotlib.dates as md
        data = values.copy()
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

        # prepare days and months axes
        fig = plt.figure(figsize=(16, 8))
        days = host_subplot(111, axes_class=AA.Axes, figure=fig)
        plt.subplots_adjust(bottom=0.1)
        months = days.twiny()

        # position months axis
        offset = -20
        new_fixed_axis = months.get_grid_helper().new_fixed_axis
        months.axis['bottom'] = new_fixed_axis(loc='bottom',
                                               axes=months,
                                               offset=(0, offset))
        months.axis['bottom'].toggle(all=True)

        # plot
        days.scatter(data.index, data)

        # formatting days axis
        if (len(data) < 400):
            # formatting days axis
            days.xaxis.set_major_locator(md.DayLocator(interval=len(data) // 30))
            days.xaxis.set_major_formatter(md.DateFormatter('%d'))
            plt.setp(days.xaxis.get_majorticklabels(), rotation=0)
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting months axis
            months.xaxis.set_major_locator(md.MonthLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%b'))
            months.set_xlim([data.index[0], data.index[-1]])
        else:
            days.xaxis.set_major_locator(md.MonthLocator(interval=len(data) // (30 * 20)))
            days.xaxis.set_major_formatter(md.DateFormatter('%b'))
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting year axis
            months.xaxis.set_major_locator(md.YearLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            months.set_xlim([data.index[0], data.index[-1]])

        plt.show()


class MakeScatterTwo(Strategy):
    def do_algorithm(self, values: pd.DataFrame, tic: str, values2: pd.DataFrame, tic2: str) -> None:
        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA
        import matplotlib.dates as md
        data = values.copy()
        data2 = values2.copy()
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

        # prepare days and months axes
        fig = plt.figure(figsize=(16, 8))
        days = host_subplot(111, axes_class=AA.Axes, figure=fig)
        plt.subplots_adjust(bottom=0.1)
        months = days.twiny()

        # position months axis
        offset = -20
        new_fixed_axis = months.get_grid_helper().new_fixed_axis
        months.axis['bottom'] = new_fixed_axis(loc='bottom',
                                               axes=months,
                                               offset=(0, offset))
        months.axis['bottom'].toggle(all=True)

        # plot
        days.scatter(data.index, data, label=tic)
        days.scatter(data.index, data2, label=tic2)
        days.legend(loc=0)
        # formatting days axis
        if (len(data) < 400):
            # formatting days axis
            days.xaxis.set_major_locator(md.DayLocator(interval=len(data) // 30))
            days.xaxis.set_major_formatter(md.DateFormatter('%d'))
            plt.setp(days.xaxis.get_majorticklabels(), rotation=0)
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting months axis
            months.xaxis.set_major_locator(md.MonthLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%b'))
            months.set_xlim([data.index[0], data.index[-1]])
        else:
            days.xaxis.set_major_locator(md.MonthLocator(interval=len(data) // (30 * 20)))
            days.xaxis.set_major_formatter(md.DateFormatter('%b'))
            days.set_xlim([data.index[0], data.index[-1]])

            # formatting months axis
            months.xaxis.set_major_locator(md.YearLocator())
            months.xaxis.set_major_formatter(md.DateFormatter('%Y'))
            months.set_xlim([data.index[0], data.index[-1]])

        plt.show()



if __name__ == "__main__":
    df = pd.read_csv("lib/stocks_open.csv", index_col=0)
    df2 = pd.read_csv("lib/stocks_volume.csv", index_col=0)

    tic = "NEL"
    tic2 = "VISTN"
    values = df.loc[tic]
    values2 = df.loc[tic2]
    values3 = df2.iloc[10:20].transpose().cumsum().iloc[df2.shape[1] - 1]
    context = Context(MakeLinePlot())

    context.make_diagram(values, tic)

    context.strategy = MakeBarDiagram()
    context.make_diagram(values3, "10 random stocks volume acuumelated over the year")

    context.strategy = MakeScatterPlot()
    context.make_diagram(values, tic)

    context.strategy = MakeHistogram()
    context.make_diagram(values, tic)

    context.strategy = MakeDetailedScatter()
    context.make_diagram(values, tic)

    context.strategy = MakeScatterPlot()
    context.make_diagram(values, tic)

    context.strategy = MakeScatterTwo()
    context.make_diagram(values, tic, values2, tic2)
