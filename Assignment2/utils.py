from operator import itemgetter
import csv

def write_submission(recommendations, submission_file):
    """
    Function which writes submission, ordered on the probability obtained by the model.
    The columns are SearchId, PropertyId and Relevance    
    """
    global rows
    submission_path = "/home/marleen/projects/DM2018/Assignment2/Bench_Results/"+submission_file
    rows = [(srch_id, prop_id, relevance)
        for srch_id, prop_id, relevance, rank_float
        in sorted(recommendations, key=itemgetter(0,3))]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("SearchId", "PropertyId", "Relevance"))
    writer.writerows(rows)
