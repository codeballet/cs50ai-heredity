import csv
import copy
import itertools
import sys
import numpy as np


PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)

    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def ancestors(people, person, d, l):
    """
    recursively build and return dictionary of ancestors
    """
    # get parents for person
    parents = get_parents(people, person)

    if len(l) == 0 and all(v is None for v in parents):
        # base case, list empty and no parents found
        return d
    elif len(l) != 0 and all(v is None for v in parents):
        # list not empty and no parents found
        # recursively call function, removing parent from list
        l_copy = copy.deepcopy(l)
        for parent in l:
            l_copy.remove(parent)
            return ancestors(people, parent, d, l_copy)
    else:
        # parents found, add to dict and list
        d[person] = parents
        l.extend(parents)
        # recursively call function, removing parent from list
        l_copy = copy.deepcopy(l)
        for parent in l:
            l_copy.remove(parent)
            return ancestors(people, parent, d, l_copy)


def ancestors_calc(people, person, one_gene, two_genes, ancestor_dict):
    print('in ancestors_calc')
    print(f'ancestor_dict: {ancestor_dict}')
    print(f'checking for person: {person}')
    ancestor_dict_copy = copy.deepcopy(ancestor_dict)
    for ancestor in ancestor_dict:
        print(f'iterating on: {ancestor}')
        ancestor_dict_copy.pop(ancestor)
        for parent in ancestor_dict[ancestor]:
            if parent not in ancestor_dict:
                # base case, ancestor has no parents
                print('ancestor has no parents')
                p_gene = parents_calc(people, person, one_gene, two_genes)
                print(f'p_gene for {ancestor}: {p_gene}')
                return p_gene
            else:
                # ancestor has parents
                print(f'mother of {ancestor}: {ancestor_dict[ancestor][0]}')
                print(f'father of {ancestor}: {ancestor_dict[ancestor][1]}')
                for parent in ancestor_dict[ancestor]:
                    return ancestors_calc(people, parent, one_gene, two_genes, ancestor_dict_copy)


def parents_calc(people, person, one_gene, two_genes):
    """
    return the probability of inheriting the gene from parents
    """
    p_gene = 0
    mother = people[person]["mother"]
    father = people[person]["father"]
    
    mother_gene_state = gene_state(mother, one_gene, two_genes)
    father_gene_state = gene_state(father, one_gene, two_genes)

    # probability of getting gene from mother
    if mother_gene_state == 0:
        p_gene_mother = PROBS["mutation"]
        p_gene_mother_not = 1 - p_gene_mother
    elif mother_gene_state == 1:
        p_gene_mother = 0.5
        p_gene_mother_not = 1 - p_gene_mother
    else:
        # mother has two genes
        p_gene_mother = 1 - PROBS["mutation"]
        p_gene_mother_not = 1 - p_gene_mother

    # probability of getting gene from father
    if father_gene_state == 0:
        p_gene_father = PROBS["mutation"]
        p_gene_father_not = 1 - p_gene_father
    elif father_gene_state == 1:
        p_gene_father = 0.5
        p_gene_father_not = 1 - p_gene_father
    else:
        # father has two genes
        p_gene_father = 1 - PROBS["mutation"]
        p_gene_father_not = 1 - p_gene_father

    # calculate probability of inheriting gene
    if person in one_gene:
        # gene either comes:
        # from mother, not from father; or not from mother, but from father
        # add up the two possibilities
        p_gene = p_gene_mother * p_gene_father_not + p_gene_mother_not * p_gene_father
    elif person in two_genes:
        # gene comes from mother and father
        p_gene = p_gene_mother * p_gene_father
    else:
        # no gene from mother, no gene from father
        p_gene = p_gene_mother_not * p_gene_father_not
        
    return p_gene


def gene_state(person, one_gene, two_genes):
    """
    return how many genes the person has
    """
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0


def get_parents(people, person):
    """
    return list of parents for a person
    if no parents, the list contains None elements
    """
    mother = people.get(person, {}).get("mother", None)
    father = people.get(person, {}).get("father", None)

    return [mother, father]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # variables for joint probability calculation
    p_joint = 1
    p_register = []

    # iterate through people
    for person in people:
        p_gene = 0
        p_trait = 0
        p_gene_trait = 0
        ancestor_dict = ancestors(people, person, {}, [])

        if len(ancestor_dict) != 0:
            # have parents, get conditional probability
            p_gene = ancestors_calc(people, person, one_gene, two_genes, ancestor_dict)
        else:
            # no parents, get unconditional probability
            if person in one_gene:
                p_gene = PROBS["gene"][1]
            elif person in two_genes:
                p_gene = PROBS["gene"][2]
            else:
                # no gene
                p_gene = PROBS["gene"][0]

        # probability of having or not having trait
        if person in one_gene:
            if person in have_trait:
                p_trait = PROBS["trait"][1][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][1][False]
        elif person in two_genes:        
            if person in have_trait:
                p_trait = PROBS["trait"][2][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][2][False]
        else:
            # no gene
            if person in have_trait:
                p_trait = PROBS["trait"][0][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][0][False]

        # calculate probability for gene and trait, then register value
        p_gene_trait = p_gene * p_trait
        p_register.append(p_gene_trait)

    # calculate entire joint probability
    for p in p_register:
        p_joint *= p

    return p_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    probabilities_list = list(probabilities.keys())

    for person in probabilities_list:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            # no gene
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            # no trait
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    # get probabilites of gene and trait as lists for each person
    probabilities_copy = copy.deepcopy(probabilities)
    for person in probabilities_copy:
        gene_list = []
        trait_list = []

        for i in range(3):
            gene_list.append(probabilities_copy[person]["gene"][i])

        trait_list.append(probabilities_copy[person]["trait"][True])
        trait_list.append(probabilities_copy[person]["trait"][False])

        # turn lists into numpy arrays
        gene_arr = np.asarray(gene_list)
        trait_arr = np.asarray(trait_list)

        # normalize arrays and turn to lists
        gene_arr = gene_arr / gene_arr.sum()
        trait_arr = trait_arr / trait_arr.sum()

        # turn normalized arrays back into lists
        gene_list = gene_arr.tolist()
        trait_list = trait_arr.tolist()

        # update probabilites with normalized values
        for i in range(3):
            probabilities[person]["gene"][i] = gene_list[i]

        probabilities[person]["trait"][True] = trait_list[0]
        probabilities[person]["trait"][False] = trait_list[1]


if __name__ == "__main__":
    main()
