# Introduction to the Artificial Intelligence Heredity

Heredity is an Artificial Intelligence that assess the likelihood of a person having a particular genetic trait.

# Background

Mutated versions of the [GJB2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1285178/) gene are one of the leading causes of hearing impairment in newborns. Each person carries two versions of the gene, so each person has the potential to possess either 0, 1, or 2 copies of the hearing impairment version GJB2. Unless a person undergoes genetic testing, though, it’s not so easy to know how many copies of mutated GJB2 a person has. This is some “hidden state”: information that has an effect that we can observe (hearing impairment), but that we don’t necessarily directly know. After all, some people might have 1 or 2 copies of mutated GJB2 but not exhibit hearing impairment, while others might have no copies of mutated GJB2 yet still exhibit hearing impairment

Every child inherits one copy of the GJB2 gene from each of their parents. If a parent has two copies of the mutated gene, then they will pass the mutated gene on to the child; if a parent has no copies of the mutated gene, then they will not pass the mutated gene on to the child; and if a parent has one copy of the mutated gene, then the gene is passed on to the child with probability 0.5. After a gene is passed on, though, it has some probability of undergoing additional mutation: changing from a version of the gene that causes hearing impairment to a version that doesn’t, or vice versa.

Given information about people, who their parents are, and whether they have a particular observable trait (e.g. hearing loss) caused by a given gene, the Heredity AI will infer the probability distribution for each person’s genes, as well as the probability distribution for whether any person will exhibit the trait in question.

# Structure for the data input to the AI

The AI reads CSV files containing data structured in columns: name, mother, father, trait.

The data should be of the types: string, string, string, integer.

As an example, the provided file `data/family0.csv` indicates that Harry has Lily as a mother, James as a father, and the empty cell for trait means we don’t know whether Harry has the trait or not. James, meanwhile, has no parents listed in the our data set (as indicated by the empty cells for mother and father), and does exhibit the trait (as indicated by the 1 in the trait cell). Lily, on the other hand, also has no parents listed in the data set, but does not exhibit the trait (as indicated by the 0 in the trait cell).

# Making inferences on the data based on statistics

The statistics that the AI uses to make inferences about populations are defined in the dictionary PROBS. PROBS contains a number of constants representing probabilities of various different events. All of these events have to do with how many copies of a particular gene a person has (hereafter referred to as simply “the gene”), and whether a person exhibits a particular trait (hereafter referred to as “the trait”) based on that gene. The data here is loosely based on the probabilities for the hearing impairment version of the GJB2 gene and the hearing impairment trait, but by changing these values, you could use your AI to draw inferences about other genes and traits as well!

# How to run the AI

The AI uses numpy for calculations, so you need to install the numpy dependency. You may use the command:

```
pip install -r requirements.txt
```

To run the program, enter:

```
python heredity.py <datafile>
```

For instance, if you want to run the program with the provided file in `data/family0.csv`, you would run:

```
python heredity.py data/family0.csv
```

# Intellectual Property Rights

MIT
