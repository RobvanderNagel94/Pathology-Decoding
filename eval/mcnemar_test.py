from statsmodels.stats.contingency_tables import mcnemar

def McNemarTest(y_true, y_pred_base, y_pred_comp):
    """
    Compare model performances between two classifier models using the McNemar's test. The test is applied to a 2×2
    contingency table.

    H0 : Model errors are significantly different.
    H1 : Model errors are not significantly different.

    Parameters
    ----------
    y_true : numpy array
         1D array of the correct binary values.

    y_pred_base : numpy array
         1D array of the predicted binary values from the base model (e.g., deep4).

    y_pred_comp : numpy array
         1D array of the predicted binary values from the model to compare.

    Returns
    -------
    p_value :  float
        p value
    statistic : float
        McNemar test statistic.

    References
    ----------
    .. https://www.statsmodels.org/devel/generated/statsmodels.stats.contingency_tables.mcnemar.html
    """

    counts_base = ['yes' if y_pred_base[i] == y_true[i] else 'no' for i in range(len(y_true))]
    counts_compare = ['yes' if y_pred_comp[i] == y_true[i] else 'no' for i in range(len(y_true))]

    # Create contingency values
    yes_yes = 0
    yes_no = 0
    no_yes = 0
    no_no = 0
    for i in range(len(counts_base)):
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'yes'):
            yes_yes += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'no'):
            no_no += 1
        if (counts_base[i] == 'yes') and (counts_compare[i] == 'no'):
            yes_no += 1
        if (counts_base[i] == 'no') and (counts_compare[i] == 'yes'):
            no_yes += 1

    # define contingency table
    contingency_table = [[yes_yes, yes_no],
                         [no_yes, no_no]]

    # McNemar's test with the continuity correction
    result = mcnemar(contingency_table, exact=False, correction=True)
    p_value = result.pvalue
    statistic = result.statistic

    print('statistic=%.3f, p-value=%.3f' % (statistic, p_value))

    # compute significance H0 and H1
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
    return statistic, p_value
