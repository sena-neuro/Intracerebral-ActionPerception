import pandas as pd
import os
import re
import pprint
pp = pprint.PrettyPrinter(indent=4)
import pickle

def get_leads_for_regions(lead_file_path, region_labels_path):
    
    lead_df = pd.read_csv(lead_file_path,sep=" ",names=["node_idx","lead_idx","region_idx"]).drop(columns="node_idx")

    # Drop duplicates created by node idx
    lead_df = lead_df.drop_duplicates()
    
    # Drop rows with a 0 value
    lead_df = lead_df[(lead_df != 0).all(1)]
    
    # Read file containing region and lead labels and idx set the df index to idx in the file
    region_lead_labes_df = pd.read_csv(region_labels_path,sep=" ",names=["idx","label"]).set_index("idx",verify_integrity=True)

    # Change every idx to the corresponding label from region_lead_labels_df
    label_region_df = lead_df.applymap(lambda x:region_lead_labes_df.loc[x].label)
    
    # Make new index
    label_region_df.reset_index(drop=True)
    
    return label_region_df

def get_lead_list(label_region_df,region_list):
    # Given a list of regions and a df that maps lead names to regions outputs a list of lead names formatted as specified
    lead_names = label_region_df.loc[label_region_df['region_idx'].isin(region_list)].lead_idx
    lead_names = lead_names.apply(lambda x: tuple(x.split('_', 2))).to_list()
    return lead_names

if __name__ == "__main__":
    # Read file containing node lead and region idx drop node idx because it is not going to be used
    left_lead_file_path = "updated_left_lead_regions.csv"
    region_lead_labes_path = "updated_left_region_lead_labels.csv"
    label_region_df=get_leads_for_regions(left_lead_file_path,region_lead_labes_path)
    region_list=["PMm_Left"]
    lead_names = get_lead_list(label_region_df,region_list)
    print(lead_names)