import subprocess

print("\t")
print("\t".join(["postediting", "question-question", "headlines", "plagiarism", "answer-answer"]))

for feat in ["bow","bodt","pwe","all"]:
    print(feat, end="\t")
    for name in ["postediting","question-question","headlines","plagiarism","answer-answer"]:
        score = subprocess.check_output("correlation-noconfidence.pl", ["../test-gs/STS2016.gs."+name+".txt"," ../output/"+feat+"/predict."+name+".txt"])
        print(score, end="\t")
    print("")