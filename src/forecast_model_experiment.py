from plotter import ForecastingPlotter, ReconciliationPlotter

def run_snp500_forecasting_experiment():
    """
    Run the S&P 500 forecasting experiment.
    """
    fins_forecasting_plotter = ForecastingPlotter(
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\snp500_results\forecasting",
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\snp500_results\forecasting\plots"
    )

    # fins_forecasting_plotter.horizon_metric_models('RMSE')
    # fins_forecasting_plotter.line_horizons_models(
    #     models=['DLM', 'ARIMA'],
    #     horizons=[0, 14, 29],
    #     limit=None,
    #     start=0
    # )
    # fins_forecasting_plotter.line_point_horizon_models(
    #     folder_path='horizons_models',
    #     horizons=None,
    #     limit=None,
    #     start=0
    # )

    # fins_forecasting_plotter.line_point_horizon_models(
    #     folder_path='horizons_models',
    #     horizons=[0, 14, 29],
    #     limit=None,
    #     start=0
    # )

    # fins_forecasting_plotter.horizons_amp_models(
    #     models=['DLM', 'ARIMA', 'LSTM', 'MLP', 'Conv1D'],
    #     start=90,
    #     std=True,
    #     show=False
    # )

    fins_forecasting_plotter.table_error_metrics(
        group=5
    )

def run_synth_forecasting_experiment():
    """
    Run the synthetic forecasting experiment.
    """
    synth_forecasting_plotter = ForecastingPlotter(
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\synth_results\forecasting",
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\synth_results\forecasting\plots"
    )

    # synth_forecasting_plotter.horizon_metric_models('RMSE')
    # synth_forecasting_plotter.line_horizons_models(
    #     models=['DLM', 'ARIMA'],
    #     horizons=[0, 14, 29],
    #     limit=60,
    #     start=0
    # )
    # synth_forecasting_plotter.line_point_horizon_models(
    #     folder_path='horizons_models',
    #     horizons=None,
    #     limit=60,
    #     start=0
    # )

    # synth_forecasting_plotter.line_point_horizon_models(
    #     folder_path='horizons_models',
    #     horizons=[0, 14, 29],
    #     limit=60,
    #     start=0
    # )

    # synth_forecasting_plotter.horizons_amp_models(
    #     models=['DLM', 'ARIMA', 'LSTM', 'MLP', 'Conv1D'],
    #     start=90,
    #     std=True,
    #     show=False
    # )

    # synth_forecasting_plotter.table_error_metrics(
    #     group=5
    # )

    synth_forecasting_plotter.table_parameters_time()

def run_synth_reconciliation_experiment():
    
    fins_plotter = ReconciliationPlotter(
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\synth_results\reconciliation", 
        r"D:\Documentos\ICAI\TFG\Code\DLM Model\synth_results\reconciliation\plots")
    

    # fins_plotter.table_times()
    # fins_plotter.models_ranking(group=6, metric='RMSE')

    # fins_plotter.errors_by_group(group=6, metric='RMSE', se=False)
    # fins_plotter.errors_by_group(group=6, metric='RMSE', se=True)

    fins_plotter.plot_covariance_comparison("mint_sample", "gaussian", 5)


if __name__ == '__main__':
    # run_snp500_forecasting_experiment()
    # run_synth_forecasting_experiment()

    run_synth_reconciliation_experiment()