import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
from math import ceil

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, cross_val_score, LearningCurveDisplay
from category_encoders.target_encoder import TargetEncoder

from hyperopt import Trials
from time import perf_counter

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input


def drop_contains(dataset: pd.DataFrame, column_to_filter: str, regex_str: str):
    """
    Retourne un dataframe sans les lignes qui contiennent le regex renseigné

    Positional arguments : 
    -------------------------------------
    dataset:  pd.DataFrame: le dataframe contenant les lignes que l'on souhaite filtrer
    column_to_filter : str : colonne du dataframe sur laquelle appliquer un filtre
    regex_str : str : chaîne de caractères recherchée
    """
    mask = ~dataset[column_to_filter].str.contains(
        regex_str, case=False, regex=True, na=False)
    subset = dataset.loc[mask]

    print(dataset.shape[0] - subset.shape[0], 'lignes supprimées, i.e.', round(
        (dataset.shape[0]-subset.shape[0])/dataset.shape[0]*100, 2), '% des données')

    return subset


def missing_values_by_column(dataset: pd.DataFrame):
    """
    Retourne un dataframe avec le nombre et le pourcentage de valeurs manquantes par colonnes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les colonnes dont on veut connaitre le pourcentage de vide
    """

    missing_values_series = dataset.isnull().sum()
    missing_values_df = missing_values_series.to_frame(
        name='Number of Missing Values')
    missing_values_df = missing_values_df.reset_index().rename(
        columns={'index': 'VARIABLES'})

    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (dataset.shape[0]) * 100, 2)

    missing_values_df = missing_values_df.sort_values(
        'Number of Missing Values')

    return missing_values_df


def plot_heatmap_correlation_matrix(correlation_matrix: pd.DataFrame, title: str, figsize: tuple, palette: str):
    """
    Affiche la matrice de corrélation sous forme de heatmap

    Positional arguments : 
    -------------------------------------
    correlation_matrix : pd.DataFrame : matrice de corrélation
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    palette : str : palette seaborn utilisée
    """

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)

    mask_upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    ax = sns.heatmap(correlation_matrix, annot=True, mask=mask_upper_triangle, vmin=-1, vmax=1, center=0,
                     cmap=sns.color_palette(palette, as_cmap=True),
                     annot_kws={"fontsize": 16, 'fontname': 'Open Sans'},
                     cbar_kws={"shrink": .5},
                     linewidth=1.5, linecolor='w', fmt='.2f', square=True)

    plt.title(title, size=20, fontname='Corbel', pad=20)

    plt.show()


def filter_outlier(dataset: pd.DataFrame, x_column: str):
    """
    Retourne un dataframe sans outliers

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données dont on souhaite extraire les outliers
    x_column : str : nom de la colonne contenant possiblement des outliers
    """

    outliers = [y for stat in boxplot_stats(
        dataset[x_column]) for y in stat['fliers']]
    mask_outliers = dataset[x_column].isin(outliers)
    subset = dataset.loc[~mask_outliers]

    return subset


def add_effectif_categ(dataset: pd.DataFrame, column_to_count: str):
    """
    Retourne un dataframe avec une colonne en plus indiquant l'effectif de chaque modalité d'une variable catégorielle
    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe à modifier
    column_to_count : str : nom de la variable catégorielle dont l'on souhaite afficher l'effectif par modalité
    """
    effectif = dataset[column_to_count].value_counts().to_dict()

    subset = dataset.copy()
    subset[column_to_count + ' (n)'] = subset.apply(lambda row: str(
        row[column_to_count]) + '\n(n={:_})'.format(effectif[row[column_to_count]]), axis=1)

    return subset


def plot_boxplot(dataset: pd.DataFrame, numeric_var: str, title: str, figsize: tuple, categ_var=None, palette='Set2'):
    """
    Affiche un graphique avec un ou plusieurs boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs dont on veut étudier la distribution
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    categ_var : str : nom de la colonne contenant les catégories 
    (si on souhaite regrouper les variables numériques par catégorie)
    palette : str or list of strings : nom de la palette seaborn utilisée ou liste de couleurs personnalisées
    """
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)
    plt.rcParams['axes.labelpad'] = '30'

    subset = add_effectif_categ(dataset, categ_var)

    ax = sns.boxplot(data=subset, x=numeric_var, y=categ_var + ' (n)',
                     orient='h', palette=palette, saturation=0.95,
                     showfliers=False,
                     medianprops={"color": "#c2ecff", 'linewidth': 3.0},
                     showmeans=True,
                     meanprops={'marker': 'o', 'markeredgecolor': 'black',
                                'markerfacecolor': '#c2ecff', 'markersize': 10},
                     boxprops={'edgecolor': 'black', 'linewidth': 1.5},
                     capprops={'color': 'black', 'linewidth': 1.5},
                     whiskerprops={'color': 'black', 'linewidth': 1.5})

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))

    plt.title(title, fontname='Corbel', color=rgb_text, fontsize=26, pad=20)
    plt.xlabel(numeric_var, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.ylabel(categ_var, fontsize=20, fontname='Corbel', color=rgb_text)
    ax.tick_params(axis='both', which='major',
                   labelsize=16, labelcolor=rgb_text)

    plt.show()


def plot_violinplot(dataset: pd.DataFrame, numeric_var: str, title: str, figsize: tuple, categ_var=None, palette='Set2'):
    """
    Affiche un graphique avec un ou plusieurs boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs dont on veut étudier la distribution
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    categ_var : str : nom de la colonne contenant les catégories 
    (si on souhaite regrouper les variables numériques par catégorie)
    palette : str or list of strings : nom de la palette seaborn utilisée ou liste de couleurs personnalisées
    """
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)

    subset = add_effectif_categ(dataset, categ_var)
    subset = filter_outlier(subset, numeric_var)
    ax = sns.violinplot(data=subset, y=numeric_var, x=categ_var +
                        ' (n)', orient='v', linewidth=2.5, width=1, palette=palette)

    plt.title(title, fontname='Corbel', color=rgb_text, fontsize=26, pad=20)
    plt.xlabel(categ_var, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.ylabel(numeric_var, fontsize=20, fontname='Corbel', color=rgb_text)
    ax.tick_params(axis='both', which='major',
                   labelsize=16, labelcolor=rgb_text)

    plt.show()


def plot_donut(dataset: pd.DataFrame, categ_var: str, palette: str, text_color: str, title: str, figsize: tuple):
    """
    Affiche un donut de la répartition d'une variable qualitative

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative

    palette : strings : nom de la palette seaborn à utiliser
    text_color : str : couleur du texte
    title : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid', palette=palette)
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontname='Corbel', fontsize=30)
        plt.rcParams.update({'axes.labelcolor': text_color, 'axes.titlecolor': text_color, 'legend.labelcolor': text_color,
                             'axes.titlesize': 16, 'axes.labelpad': 10})

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = ax.pie(pie_series, labels=pie_series.index, autopct='%.0f%%', pctdistance=0.85,
                                       textprops={
                                           'fontsize': 20, 'color': text_color, 'fontname': 'Open Sans'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    ax.axis('equal')
    ax.add_artist(plt.Circle((0, 0), 0.7, fc='white'))

    plt.tight_layout()
    plt.show()


def build_score_df_from_cv(cv_results: dict, sets_name: [str]):
    """
    Retourne un dataframe contenant des scores de régression à partir de l'output d'une validation croisée sklearn

    Positional arguments : 
    -------------------------------------
    cv_results : dict : output de la méthode cross_validate (sklearn.model_selection.cross_validate)
    sets_name : list of strings : nom des jeux de données à inclure dans le dataframe. ex : ['train', 'test']
    """
    score_data = []
    for set_name in sets_name:
        rmse = abs(
            cv_results[set_name + '_neg_root_mean_squared_error'].mean())
        mape = abs(
            cv_results[set_name + '_neg_mean_absolute_percentage_error'].mean())
        r2 = cv_results[set_name + '_r2'].mean()
        mse = abs(cv_results[set_name + '_neg_mean_squared_error'].mean())
        rmsle = abs(
            cv_results[set_name + '_neg_mean_squared_log_error'].mean())
        mae = abs(cv_results[set_name + '_neg_mean_absolute_error'].mean())
        time = cv_results['fit_time'].mean()

        rmse_std = abs(
            cv_results[set_name + '_neg_root_mean_squared_error'].std())
        rmsle_std = abs(
            cv_results[set_name + '_neg_mean_squared_log_error'].std())
        mape_std = abs(
            cv_results[set_name + '_neg_mean_absolute_percentage_error'].std())
        r2_std = cv_results[set_name + '_r2'].std()
        mse_std = abs(cv_results[set_name + '_neg_mean_squared_error'].std())
        mae_std = abs(cv_results[set_name + '_neg_mean_absolute_error'].std())

        score_data.append([set_name, rmsle, rmse, mape, r2, mse, mae,
                          time, rmsle_std, rmse_std, mape_std, r2_std, mse_std, mae_std])

    score_df = pd.DataFrame(data=score_data, columns=['set', 'rmsle', 'rmse', 'mape', 'r2', 'mse', 'mae', 'fit_time', 
                                                      'rmsle_std', 'rmse_std', 'mape_std', 'r2_std', 'mse_std', 'mae_std'])

    return score_df


def make_preprocessor(transformers: [dict]):
    """
    Retourne un objet preprocessor contenant les transformations à appliquer aux données avant l'entrainement du modèle

    Positional arguments : 
    -------------------------------------
    transformers : list of tuples : liste des tuples (modifications à appliquer 
    categorical_features : list of strings : liste des variables catégorielles à transformer (one hot encoder)
    """
    steps = []
    for transformer in transformers:
        pipeline = make_pipeline(*transformer['estimator'])
        steps.append((pipeline, transformer['feature']))

    preprocessor = make_column_transformer(*steps)

    return preprocessor


def cross_validate_and_score(regressors: [dict], X: np.array, y: np.array, features: [dict], cv=5):
    """
    Retourne un dataframe contenant les scores moyens mesurant la performance 
    de modèles de régression après cross-validation et la liste des modèles entrainés. 

    Positional arguments : 
    -------------------------------------
    regressors : list of dict : liste des modèles (sous forme de dictionnaire) 
    X : np.array : observations 
    y : np.array : cibles 
    features : dict : dictionnaire spécifiant les variables numériques et catégorielles (pour transformation avant entrainement)

    Optional arguments : 
    -------------------------------------
    cv : float : nombre de folds
    """
    scores_all_models = pd.DataFrame()
    preprocessor = make_preprocessor(features)

    for model in regressors:
        if model['with_pipeline']:
            regressor = make_pipeline(preprocessor, model['regressor'])
        else:
            regressor = model['regressor']

        cv_results = cross_validate(TransformedTargetRegressor(regressor, func=np.log, inverse_func=np.exp),
                                    X, y, cv=cv,
                                    scoring=('neg_mean_squared_log_error',
                                             'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error', 'r2',
                                             'neg_mean_squared_error', 'neg_mean_absolute_error'),
                                    n_jobs=-1,
                                    verbose=0,
                                    return_train_score=True)

        scores_cv = build_score_df_from_cv(cv_results, ['test', 'train'])
        scores_cv.insert(0, 'model', model['name'])

        scores_all_models = pd.concat(
            [scores_all_models, scores_cv], ignore_index=True)

    return scores_all_models


def count_outliers(dataset: pd.DataFrame, features: [str], whiskers=1.5):
    """
    Affiche le nombre de valeurs extremes dans le jeu de données pour les variables choisies selon la méthode des boxplots 
    (i.e. valeurs en dehors de [Q1 - whiskers*IQ ; Q3 + whiskers*IQ] avec IQ = Q3-Q1) et retourne les outliers

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    features : list of strings : liste des variables sur lesquelles filtrer les outliers

    Optional arguments : 
    -------------------------------------
    whiskers : float : position des moustaches, les données en dehors des moustaches sont considérées comme des outliers
    """
    masks = []
    for feature in features:
        outliers = [y for stat in boxplot_stats(
            dataset[feature], whis=whiskers) for y in stat['fliers']]
        masks.append(dataset[feature].isin(outliers))

    mask_outliers_all = np.logical_or.reduce(masks)
    outliers_n = dataset.loc[mask_outliers_all].shape[0]

    print('Il y a {} outliers, soit {:.2f}% du jeu de données'.format(
        outliers_n, (outliers_n/dataset.shape[0])*100))

    return dataset.loc[mask_outliers_all]


def drop_outliers_by_(dataset: pd.DataFrame, x_column: [str], whiskers=1.5):
    """
    Retourne un dataframe sans outliers

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données dont on souhaite extraire les outliers
    x_column : list of strings : nom des colonnes contenant possiblement des outliers

    Optional arguments : 
    -------------------------------------
    whiskers : float : position des moustaches, les données en dehors des moustaches sont considérées comme des outliers
    """
    masks = []
    for column in x_column:
        outliers = [y for stat in boxplot_stats(
            dataset[column], whis=whiskers) for y in stat['fliers']]
        masks.append(dataset[column].isin(outliers))

    mask_outliers_all = np.logical_or.reduce(masks)
    subset = dataset.loc[~mask_outliers_all]

    return subset


def build_xi_table(contingence_table: pd.DataFrame):
    """
    Retourne un dataframe (contenant la contribution à la non-indépendance 
    pour chaque case du tableau de contingence) et la statistique du chi-2 associée

    Positional arguments : 
    -------------------------------------
    contingence_table : pd.DataFrame : tableau de contingence
    """

    distribution_marginale_x = contingence_table.loc[:, ['Total']]
    distribution_marginale_y = contingence_table.loc[['Total'], :]

    independance_table = distribution_marginale_x.dot(
        distribution_marginale_y) / contingence_table['Total'].loc['Total']

    xi_ij = (contingence_table-independance_table)**2 / independance_table
    xi_n = xi_ij.sum().sum()
    xi_table = xi_ij/xi_n

    return xi_table, xi_n


def plot_heatmap(data: pd.DataFrame, vmax: float, titles: dict, figsize: tuple, 
                 fmt: str, annotation=True, vmin=0.0, palette="rocket_r", square=False):
    """
    Affiche une heatmap 

    Positional arguments : 
    -------------------------------------
    data : pd.DataFrame : jeu de données contenant les valeurs pour colorer la heatmap
    vmax : float : valeur maximale de l'échelle des couleurs
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'blabla'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    fmt : str : format annotations 

    Optional arguments : 
    -------------------------------------
    annotation : bool or pd.DataFrame : valeurs à afficher dans les cases de la heatmap - True : utilise data 
    vmin : float : valeur minimale de l'échelle des couleurs
    palette : str : couleurs de la heatmap
    square : bool : affiche les cases de la heatmap en carré
    """
    plt.figure(figsize=figsize)

    with sns.axes_style('white'):
        ax = sns.heatmap(data, annot=annotation, vmin=vmin, vmax=vmax, cmap=sns.color_palette(palette, as_cmap=True),
                         annot_kws={"fontsize": 16, 'fontname': 'Open Sans'}, 
                         linewidth=1, linecolor='w', fmt=fmt, square=square)

    if fmt == 'd':
        for t in ax.texts:
            t.set_text('{:_}'.format(int(t.get_text())))
    plt.title(titles['chart_title'], size=28, fontname='Corbel', pad=40)
    plt.xlabel(titles['x_title'], fontname='Corbel', fontsize=24, labelpad=20)
    ax.xaxis.set_label_position('top')
    plt.ylabel(titles['y_title'], fontname='Corbel', fontsize=24, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14,
                    labeltop=True,  labelbottom=False)

    plt.show()


def display_distribution_log(dataset: pd.DataFrame, feature: str):
    """
    Affiche deux histrogrammes : la distribution de la variable et la distribution du log de la variable

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    feature : str : nom de la variable à afficher
    """
    data = dataset.copy()
    data['Log' + feature] = np.log(data[feature])

    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='husl')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.tight_layout()

    fig.suptitle('Distribution target vs log(target)',
                 fontname='Corbel', fontsize=20, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=0.8, wspace=0.1, hspace=0.5)

    sns.histplot(data, x=feature, ax=axes[0])
    sns.histplot(data, x='Log' + feature, ax=axes[1])

    axes[0].set_title(feature, fontname='Corbel', fontsize=16, color=rgb_text)
    axes[1].set_title('log({})'.format(feature),
                      fontname='Corbel', fontsize=16, color=rgb_text)

    for i in range(2):
        axes[i].tick_params(axis='both', which='major',
                            labelsize=14, labelcolor=rgb_text)
        axes[i].grid(False, axis='x')

    plt.show()


def display_distribution(dataset: pd.DataFrame, numeric_features: [str], column_n: int, 
                         figsize: tuple, top=0.85, wspace=0.2, hspace=1.8):
    """
    Affiche la distribution de chaque variable de la liste.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    numeric_features : list of strings : liste des variables numériques dont on souhaite afficher la distribution
    column_n : int : nombre de graphique à afficher par ligne
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure 
    (ex: 0.9 -> le haut des graphiques commence à 10% de la hauteur de la figure)
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='Set2')

    fig = plt.figure(figsize=figsize)
    fig.tight_layout()

    fig.suptitle('Distribution variables numériques',
                 fontname='Corbel', fontsize=20, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    for i, feature in enumerate(numeric_features):
        sub = fig.add_subplot(
            ceil(len(numeric_features)/column_n), column_n, i + 1)
        sub.set_xlabel(feature, fontsize=14, fontname='Corbel', color=rgb_text)
        sub.set_title(feature, fontsize=16, fontname='Corbel', color=rgb_text)

        sns.histplot(dataset, x=feature)
        sub.grid(False, axis='x')
        sub.tick_params(axis='both', which='major',
                        labelsize=14, labelcolor=rgb_text)

    plt.show()


def display_barplot(dataset: pd.DataFrame, x_column: str, y_column: str, titles: dict, 
                    figsize: tuple, hue=None, legend=False, palette=None):
    """
    Affiche un barplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    x_column : str : nom de la variable à mettre sur l'axe des abscisses
    y_column : str : nombre de la variable à mettre sur l'axe des ordonnées
    titles : dict : dictionnaire contenant les titres à afficher sur le graphique 
    {"title": 'titre du graph', 'xlabel' : 'titre de l'axe des abscisses'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    hue : str : nom de la colonne à utiliser pour colorer les barres
    legend : bool : si True,  affiche la légende
    palette : str : couleurs à utiliser
    """
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=x_column, y=y_column, hue=hue,
                     data=dataset, palette=palette)

    plt.title(titles['title'], size=25, fontname='Corbel', pad=40)
    plt.ylabel(y_column, fontsize=20, fontname='Corbel')
    plt.xlabel(titles['xlabel'], fontsize=20, fontname='Corbel')
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()

    if legend:
        plt.legend(title=hue, fontsize=20, title_fontsize=20)

    plt.show()


def plot_test_vs_pred(model, y_test: np.array, X_test: np.array, target: str, label: str, text_pos: float):
    """
    Affiche un nuage de points ayant pour coordonnées (valeur prédite, vraie valeur) et la première bissectrice

    Positional arguments : 
    -------------------------------------
    model : : modèle de régression entrainé 
    y_test : np.array : cibles du jeu de données test
    X_test : np.array : jeu de données test
    target : str : nom de la colonne à prédire
    label : str : nom de la cible à afficher sur l'axe des abscisses et des ordonnées ex: label = consommation
    text_pos : float : position sur l'axe des abscisses de l'étiquette contenant le R2 de test 
    """
    y_df = pd.DataFrame(y_test).rename(columns={target: 'y_test'})
    y_df['y_pred'] = model.predict(X_test)
    r2 = r2_score(y_df['y_test'], y_df['y_pred'])

    sns.set_theme(style='white')
    plt.figure(figsize=(15, 6))
    sns.scatterplot(y_df, x='y_pred', y='y_test', alpha=0.5)
    max_plot = round(max(y_df['y_pred'].max(), y_df['y_test'].max()), 0)
    plt.plot([0, max_plot], [0, max_plot], color='coral')

    plt.text(text_pos, max_plot, '\n R2: {:.4f} \n'.format(r2), fontsize=15,
             verticalalignment='top', horizontalalignment='right',
             bbox={'pad': 0, 'boxstyle': 'round',
                   'facecolor': 'none', 'edgecolor': 'black'},
             style='normal', fontname='Open Sans')

    plt.xlabel(label + ' prédites', fontname='Corbel', fontsize=20)
    plt.ylabel('vraies ' + label, fontname='Corbel', fontsize=20)
    plt.title('Valeurs prédites vs vraies valeurs',
              fontname='Corbel', fontsize=25)

    plt.show()


def plot_top_features_linear_model(pipeline: Pipeline, X: np.array, title: str, figsize: tuple, top_n=10):
    """
    Affiche les variables ayant la plus grande importance dans un modèle linéaire

    Positional arguments : 
    -------------------------------------
    pipeline : Pipeline : pipeline contenant le modèle de régression
    X : np.array : observations
    title : str : titre du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    top_n : int : nombre de variables à afficher
    """

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)

    coeff = pipeline[1].coef_

    color_list = sns.color_palette("Set2", X.shape[1])

    idx = np.argsort(np.abs(coeff))[::-1]
    ax = plt.barh(X.columns[idx[:top_n]][::-1], coeff[idx[:top_n]][::-1])

    for i, bar in enumerate(ax):
        bar.set_color(color_list[idx[:top_n][::-1][i]])
        plt.box(False)

    plt.suptitle(" Top des Coefficients " + title,
                 fontsize=20, fontname='Corbel')

    plt.show()


def plot_feature_importance_tree_model(tree_models: [dict], features: [str], figsize: tuple, top_n=10):
    """
    Affiche les variables ayant la plus grande importance dans un ou plusieurs 
    modèles ensemblistes utilisant des arbres de décision

    Positional arguments : 
    -------------------------------------
    tree_models : list of dictionnaries : liste des modèles à analyser
    features : list of strings : liste des variables
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    top_n : int : nombre de variables à afficher
    """

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)
    color_list = sns.color_palette("Set2", len(features))

    fig, axs = plt.subplots(1, len(tree_models),
                            figsize=figsize, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.5, wspace=0.8, top=0.8)
    axs = axs.ravel()
    for i in range(len(tree_models)):
        feature_importance = tree_models[i]["model"].feature_importances_
        indices = np.argsort(feature_importance)
        indices = indices[-top_n:]

        bars = axs[i].barh(
            range(len(indices)), feature_importance[indices], color='b', align='center')
        axs[i].set_title(tree_models[i]["name"], fontsize=20)

        plt.sca(axs[i])
        plt.yticks(range(len(indices)), [features[j]
                   for j in indices], fontsize=14)

        for i, ticklabel in enumerate(plt.gca().get_yticklabels()):
            ticklabel.set_color(color_list[indices[i]])

        for i, bar in enumerate(bars):
            bar.set_color(color_list[indices[i]])
        plt.box(False)

    plt.suptitle("Top des variables les plus 'importantes'",
                 fontsize=25, fontname='Corbel')

    plt.show()


def build_model(regressor, transformers: [dict]):
    """
    Construit un modèle avec un pipeline de transformations et passe la cible au logarithme

    Positional arguments : 
    -------------------------------------
    regressor :  : modèle de régression
    transformers : list of dict : liste des transformations à appliquer sur les variables avant l'entrainement
    """
    preprocessor = make_preprocessor(transformers)
    pipeline = make_pipeline(preprocessor, regressor)
    model = TransformedTargetRegressor(
        pipeline, func=np.log, inverse_func=np.exp)

    return model


def score_best_model(regressor, transformers: [dict], best_params: dict, datasets: dict, cv=5):
    """
    Affiche R2 sur jeu de test/train + temps d'entrainement du modèle optimisé et retourne le modèle optimisé entrainé

    Positional arguments : 
    -------------------------------------
    regressors :  : modèle de régression optimisé
    transformers : list of dict : liste des transformations à appliquer sur les variables avant l'entrainement
    best_params : dict : dictionnaire des paramètres avec lesquels configurer le modèle optimisé
    datasets : dict : dictionnaire contenant les jeux de données de test et d'entrainement

    Optional arguments : 
    -------------------------------------
    cv : int : nombre de folds dans la validation croisée utilisée pour calculer 
    l'écart type du R2 (pour voir la stabilité du modèle)
    """
    model_opt = build_model(regressor, transformers)
    std = cross_val_score(
        model_opt, datasets['X_train'], datasets['y_train'], cv=cv, scoring="r2", n_jobs=-1).std()

    t1_opt_start = perf_counter()
    model_opt.fit(datasets['X_train'], datasets['y_train'])
    t1_opt_stop = perf_counter()

    best_param_df = pd.DataFrame(best_params.items(), columns=[
                                 'Param', 'Best Param'])
    display(best_param_df)

    print("Modèle optimisé : ")
    print("Test R2 : {:.4f}".format(model_opt.score(
        datasets['X_test'], datasets['y_test'])))
    print("Temps d'entrainement : {:.3f}".format(t1_opt_stop - t1_opt_start))
    print("Train R2 : {:.4f}".format(model_opt.score(
        datasets['X_train'], datasets['y_train'])))

    print(
        "\nEcart type R2 (validation croisée sur train set) : {:.4f}".format(std))

    return model_opt


def build_trial_df(trials: Trials, loss: str):
    """
    Retourne un dataframe contenant des informations sur les itérations de 
    l'optimisation réalisée avec hyperopt (score, paramètres testés)

    Positional arguments : 
    -------------------------------------
    trials : hyperopt.Trials : objet Trials contenant les informations sur chaque itération 
    loss : str : score à minimiser lors de l'optimisation
    """
    trials_df = pd.DataFrame(
        [pd.Series(t["misc"]["vals"]).apply(lambda row: row[0]) for t in trials])
    trials_df[loss] = [t["result"]["loss"] for t in trials]
    trials_df["trial_number"] = trials_df.index

    return trials_df


def display_lineplot(dataset: pd.DataFrame, x_column: str, y_column: str, 
                     figsize: tuple, titles: dict, grid_x=True, palette='husl'):
    """
    Affiche un graphique représentant l'évolution du score à chaque itération

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : tableau contenant les scores et les itérations 
    x_column : str : nom de la colonne contenant les itérations
    y_column : str : nom de la colonne contenant les scores
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles : dict : titres du graphique

    Optional arguments : 
    -------------------------------------
    grid_x : bool : si True affiche la grille de l'axe des abscisses
    palette : str : nom de la palette seaborn à utiliser
    """
    sns.set_theme(style='whitegrid', palette=palette)
    plt.figure(figsize=figsize)
    sns.lineplot(dataset, x=x_column, y=y_column)
    plt.title(titles['title'], fontname='Corbel', fontsize=30, pad=30)
    plt.ylabel(titles['ylabel'], fontname='Corbel', fontsize=20)
    plt.xlabel(titles['xlabel'], fontname='Corbel', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)

    if not grid_x:
        plt.grid(False, axis='x')

    plt.show()


def display_scatter_by(dataset: pd.DataFrame, column_category: str, 
                       x_column: str, y_column: str, column_n: int, titles: dict, figsize: tuple,
                       top=0.91, wspace=0.2, hspace=0.9):
    """
    Affiche un nuages de points pour chaque modalité d'une catégorie

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    column_category : str : nom de la colonne contenant la catégorie
    x_column : str : nom de la colonne contenant la variable à afficher sur l'axe des abscisses
    y_column : str : nom de la colonne contenant la variable à afficher sur l'axe des ordonnées
    column_n : int : nombre de graphiques à afficher sur une ligne
    titles : dict : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure 
    (ex: 0.9 -> le haut des graphiques commence à 10% de la hauteur de la figure)
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid')

    fig = plt.figure(figsize=figsize)
    fig.tight_layout()

    fig.suptitle(titles['title'], fontname='Corbel',
                 fontsize=40, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    categories = dataset[column_category].unique()

    for i, category in enumerate(categories):
        sub = fig.add_subplot(ceil(len(categories)/column_n), column_n, i + 1)
        sub.set_xlabel(titles['xlabel'], fontsize=14,
                       fontname='Corbel', color=rgb_text)
        sub.set_title(category, fontsize=16, fontname='Corbel', color=rgb_text)

        mask = dataset[column_category] == category
        sns.scatterplot(dataset.loc[mask], x=x_column, y=y_column)
        sub.tick_params(axis='both', which='major',
                        labelsize=14, labelcolor=rgb_text)

    plt.show()


def test_transformer(dataset: pd.DataFrame, vars_to_transform: [str], 
                     transformers: list, transformer_name: str, figsize: tuple, top=0.9, wspace=0.1, hspace=0.7):
    """
    Affiche la distribution des variables choisies avant et après transformation

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    vars_to_transform : list of str : liste des variables dont on souhaite transformer la distribution
    transformers : list : liste des objects de transformation (ex: StandardScaler)
    transformer_name : str : nom de la transformation appliquée à afficher sur le graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure 
    (ex: 0.9 -> le haut des graphiques commence à 10% de la hauteur de la figure)
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    df_transform = dataset.copy()
    for transformer in transformers:
        df_transform[vars_to_transform] = transformer.fit_transform(
            df_transform[vars_to_transform].values)

        if transformer_name == 'Yeo-Johnson':
            lbd = transformer.lambdas_

    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='Set2')

    fig, axes = plt.subplots(len(vars_to_transform), 2, figsize=figsize)
    fig.tight_layout()

    fig.suptitle('Distribution avant et après transformation ({})'.format(
        transformer_name), fontname='Corbel', fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=top, wspace=wspace, hspace=hspace)

    if len(vars_to_transform) > 1:
        for i, feature in enumerate(vars_to_transform):
            sns.histplot(dataset, x=feature, ax=axes[i, 0])
            sns.histplot(df_transform, x=feature, ax=axes[i, 1])

            axes[i, 0].set_title(feature + ' avant transformation (skewness : {:.4f})'.format(dataset[feature].skew()),
                                 fontname='Corbel', fontsize=20, color=rgb_text)

            if transformer_name == 'Yeo-Johnson':
                text_after = feature + \
                    ' après transformation (skewness : {:.4f} , lambda : {:.4f})'.format(
                        df_transform[feature].skew(), lbd[i])
            else:
                text_after = feature + \
                    ' après transformation (skewness : {:.4f})'.format(
                        df_transform[feature].skew())

            axes[i, 1].set_title(
                text_after, fontname='Corbel', fontsize=20, color=rgb_text)

            for j in range(2):
                axes[i, j].tick_params(
                    axis='both', which='major', labelsize=16, labelcolor=rgb_text)
                axes[i, j].grid(False, axis='x')

    elif len(vars_to_transform) == 1:

        sns.histplot(dataset, x=vars_to_transform[0], ax=axes[0])
        sns.histplot(df_transform, x=vars_to_transform[0], ax=axes[1])

        axes[0].set_title(vars_to_transform[0] + ' avant transformation (skewness : {:.4f})'.format(dataset[vars_to_transform[0]].skew()),
                          fontname='Corbel', fontsize=20, color=rgb_text)
        axes[1].set_title(vars_to_transform[0] + ' après transformation (skewness : {:.4f})'.format(df_transform[vars_to_transform[0]].skew()),
                          fontname='Corbel', fontsize=20, color=rgb_text)

        for j in range(2):
            axes[j].tick_params(axis='both', which='major',
                                labelsize=16, labelcolor=rgb_text)
            axes[j].grid(False, axis='x')

    plt.show()


def display_learning_curve(regressor, X_train: np.array, y_train: np.array, transformers: dict, figsize: tuple, cv=5):
    """
    Affiche la courbe d'apprentissage

    Positional arguments : 
    -------------------------------------
    regressor :  : modèle de régression
    X_train : np.array : jeu d'entrainement
    y_train : np.array : cibles du jeu d'entrainement
    transformers : dict : transformations à appliquer aux variables avant entrainement
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    cv : int : nombre de folds dans la validation croisée (par défaut)
    """
    sns.set_theme(style='white', palette='Set2')
    fig, ax = plt.subplots(figsize=figsize)

    LearningCurveDisplay.from_estimator(build_model(regressor, transformers), 
                                        X_train, y_train, scoring='r2', score_name='R2', score_type='both', cv=cv,
                                        line_kw={"marker": "o"}, ax=ax)

    plt.title('Courbe d\'apprentissage',
              fontname='Corbel', fontsize=20, pad=20)
    plt.show()


def plot_empirical_distribution(column_to_plot: pd.Series, color: tuple, titles: dict, figsize: tuple, vertical=True):
    """
    Affiche un histogramme de la distribution empirique de la variable choisie

    Positional arguments : 
    -------------------------------------
    column_to_plot : np.array : valeurs observées
    color : tuple : couleur des barres de l'histogramme
    titles : dict : titres du graphique et des axes - 
    ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'blabla'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    vertical : bool : True pour afficher l'histogramme à la verticale, False à l'horizontale
    """

    plt.figure(figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]

    with sns.axes_style('white'):
        if vertical:
            ax = sns.histplot(column_to_plot, stat="percent", discrete=True,
                              shrink=.9, edgecolor=color, linewidth=3, alpha=0.4, color=color)
            ax.set(yticklabels=[])
            sns.despine(left=True)
        else:
            ax = sns.histplot(y=column_to_plot, stat="percent", discrete=True,
                              shrink=.6, edgecolor=color, linewidth=3, alpha=0.4, color=color)
            ax.set(xticklabels=[])
            sns.despine(bottom=True)

    for container in ax.containers:
        ax.bar_label(container, size=18, fmt='%.1f%%',
                     fontname='Open Sans', padding=5)

    plt.title(titles['chart_title'], size=24,
              fontname='Corbel', pad=40, color=rgb_text)
    plt.ylabel(titles['y_title'], fontsize=20,
               fontname='Corbel', color=rgb_text)
    ax.set_xlabel(titles['x_title'], rotation=0, labelpad=20,
                  fontsize=20, fontname='Corbel', color=rgb_text)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.show()


def build_frequency_df_with_thresh(dataset: pd.DataFrame, column_to_count: str, thresh: float, other_label: str):
    """
    Retourne un dataframe avec la fréquence empirique de chaque modalité et 
    regroupe les modalités peu représentées (i.e. fréquence < limite choisie)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant la variable à étudier
    column_to_count : str : nom de la colonne contenant la variable à étudier
    thresh : float : fréquence limite en dessous de laquelle les modalités peu représentées sont regroupées en une seule classe
    other_label : str : nom de la nouvelle modalité (qui regroupe les modalités peu représentées)
    """
    frequency_df = dataset[[column_to_count]].copy()
    effectifs = dataset[column_to_count].value_counts(normalize=True).to_dict()
    frequency_df['frequency'] = frequency_df.apply(
        lambda row: effectifs[row[column_to_count]], axis=1)

    other = other_label.format(str(thresh * 100))

    frequency_df[column_to_count] = frequency_df.apply(lambda row: other if (
        row['frequency'] < thresh) else row[column_to_count], axis=1)
    frequency_df = frequency_df.sort_values('frequency', ascending=False)

    return frequency_df