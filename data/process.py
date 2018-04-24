# -*- coding: utf-8 -*-

import pickle,os
path=""
def load(file_name):
  return pickle.load(open(os.path.join(path, file_name), 'rb'))

answers=load("original/answers")
print ("have %d answers" % len(answers))

vocabulary=load("original/vocabulary")
print ("have %d words" % len(vocabulary))


with open("data/answers.txt","w") as f:
    for ans in answers:
        f.write(" ".join([ str(num) for num in answers[ans]]) + "\n")
        
with open("data/answers.txt","w") as f:
    for ans in answers:
        f.write(" ".join([ vocabulary[num] for num in answers[ans]]) + "\n")       
with open("data/vocabulary.txt","w") as f:
    f.write("\n".join(vocabulary.values()))
    
with open("insuranceQA/train") as f, open("data/train.txt","w") as out:
    for line in f:
        tokens = line.split()
        newline=""
        
        q= tokens[2]
        newline=newline + " ".join([item for item in q.split("_") if item!= "<a>"]) +"\t"
        
        a= tokens[3]
        newline=newline + " ".join( [item for item in a.split("_") if item!= "<a>"]) +"\n"
        out.write(newline)
        

with open("data/validate.txt","w") as f:
    for i,dataset in enumerate(["dev","test1","test2"]):    
    
        test1=load("original/"+dataset)
        
        for item in test1:
    		
            question=item["question"]
            bad=" ".join([str(num) for num in item["bad"]])
            good=" ".join([str(num) for num in item["good"]])
    
            q_words= " ".join([ vocabulary[w]  for w in question])
            line = "\t".join( [str(i), q_words ,good,bad])  
            f.write(line+"\n")
            
