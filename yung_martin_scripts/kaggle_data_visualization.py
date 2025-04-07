import os


from utils.utils import load_data_pandas
from visualization import plot_histogram, print_unique_values, plot_raw_trajectory
from MV_analysis import visualize_MV_barplot, visualize_MV_matrix


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data/128_fishing_trajs.csv')
    df = load_data_pandas(data_path)
    print(df.isna().sum())

    # visualize_MV_matrix(df)
    # duplicates = df[df.duplicated()]
    # print(duplicates)
    # print(df['time_gap'].max())
    # plot_histogram(
    #     df=df,
    #     feature='time_gap',
    #     num_bins=15,
    #     max_value=50,
    #     with_target=False
    # )

    print(df.describe())
    print(df.info())
    # vessel_list = print_unique_values(df=df, target_feature='label', 
                        # label='01-sailing', unique_feature='id')
    
    # for vessel in vessel_list:
    #     plot_raw_trajectory(
    #         df=df,
    #         mmsi=vessel,
    #         id_feature='id',
    #         speed_feature='euc_speed',
    #         time_feature='t',
    #         lat_feature='latitude',
    #         long_feature='longitude',
    #         num_hours='full',
    #         color_speed=False,
    #         norm='ALL',
    #         target_feature='label',
    #         label='01-sailing',
    #         scale_factor=0.5

    #     )



if __name__ == '__main__':
    main()