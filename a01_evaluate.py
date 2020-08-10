# from a01 import order
# from _up199300242up201911639 import order
# from _up199502863up201900679 import order
# from _up200808441up201908482 import order
# from _up201005341up201909212 import order
# from _up201404203up201804301 import order
from _up201603713 import order
# from _up201606649 import order
# from _up201800355 import order
# from _up201804298up200102207 import order
# from _up201908477 import order
# from _up201908581up201909146 import order
# from _up201910711 import order


def evaluate(past_moves, past_sales, demand, gain, loss, order_fn):
    """ evaluate student's moves; parameters:
        * past_moves: history of past inventories
        * past_sales: history of past sales
        * demand: true future demand (unknown to students)
        * gain: profit per sold unit
        * loss: deficit generated per unit unsold
        * order_fn: function implementing student's method
    """
    moves = []

    def market(move):
        """ demand function ("censored", as it is limited by 'move'); parameter:
            * move: quantity available for selling (i.e., inventory)
        """
        global nmoves
        if nmoves >= len(demand):
            return None
        moves.append(move)
        sales = min(move, demand[nmoves])
        nmoves += 1
        return sales

    profit = 0
    n = len(demand)
    orders = []
    sales = []
    order_fn(past_moves, past_sales, market)

    for i in range(n):
        if moves[i] > demand[i]:
            profit += demand[i] * gain - (moves[i] - demand[i]) * loss
        else:
            profit += moves[i] * gain
        print(f"{i+1}\t{demand[i]}\t{moves[i]}\t{moves[i]-demand[i]}\t{profit}")
    return profit


if __name__ == "__main__":
    import sys

    stud = sys.argv[1]
    exec(f"from {stud} import order")
    gain = 1
    loss = 9
    nmoves = 0
    order_fn = order
    import pandas as pd

    df = pd.read_csv("data-A01-2020.csv", sep="\t")
    # df = pd.read_csv("data-A01ii-2020.csv.gz", sep="\t")
    past_moves = df["Inventory"].values
    past_sales = df["Sales"].values
    df = pd.read_csv("hidden_demand-A01-2020.csv", sep="\t")
    # df = pd.read_csv("hidden_demandii-A01-2020.csv.gz", sep="\t")
    future_demand = df["Demand"].values

    profit = evaluate(past_moves, past_sales, future_demand, gain, loss, order_fn)
    print("profit", profit)