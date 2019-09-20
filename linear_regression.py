{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris = load_iris()\n",
    "df = pd.DataFrame(iris['data'], columns =iris['feature_names'])\n",
    "df = pd.DataFrame(df)\n",
    "x = df['sepal length (cm)']\n",
    "y = df['sepal width (cm)']\n",
    "x = np.array(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0xe5e4ab0>]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x = x.reshape(n,1)\n",
    "reg = LinearRegression()\n",
    "reg = reg.fit(x,y)\n",
    "y_predicted = reg.predict(x)\n",
    "plt.scatter(x,y , c ='g')\n",
    "plt.plot(x,y_predicted, c='r')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
