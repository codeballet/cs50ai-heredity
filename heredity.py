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
                print(f'joint probability: {p}')
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


# def mothers(people, person, ancestor_list):
#     # recursively get mothers
#     new_mother = people.get(person, {}).get("mother", None)
#     if new_mother == None:
#         # base case
#         return ancestor_list
#     ancestor_list.append(new_mother)
#     return mothers(people, new_mother, ancestor_list)


# def fathers(people, person, ancestor_list):
#     # recursively get fathers
#     new_father = people.get(person, {}).get("father", None)
#     if new_father == None:
#         # base case
#         return ancestor_list
#     ancestor_list.append(new_father)
#     return fathers(people, new_father, ancestor_list)


def ancestors(people, person, ancestor_dict):
    # get ancestors two generations back
    mother = people.get(person, {}).get("mother", None)
    father = people.get(person, {}).get("father", None)
    mother_mother = people.get(mother, {}).get("mother", None)
    mother_father = people.get(mother, {}).get("father", None)
    father_mother = people.get(father, {}).get("mother", None)
    father_father = people.get(father, {}).get("father", None)

    # lists of ancestors
    parents = [mother, father]
    mother_grandparents = [mother_mother, mother_father]
    father_grandparents = [father_mother, father_father]

    if all(v is None for v in parents):
        # no parents found
        return False

    # add parents to dictionary
    ancestor_dict["mother"] = mother
    ancestor_dict["father"] = father

    # check for grandparents
    if not all(v is None for v in mother_grandparents):
        # mother has parents, add to dict
        ancestor_dict["mother_mother"] = mother_mother
        ancestor_dict["mother_father"] = mother_father

    if not all(v is None for v in father_grandparents):
        # father has parents, add to dict
        ancestor_dict["father_mother"] = father_mother
        ancestor_dict["father_father"] = father_father

    return ancestor_dict


def gene_state(person, one_gene, two_genes):
    if person in one_gene:
        return 1
    elif person in two_genes:
        return 2
    else:
        return 0


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
    print(f'people: {people}')
    print(f'one_gene: {one_gene}')
    print(f'two_genes: {two_genes}')
    print(f'have_trait: {have_trait}')

    # variables for joint probability calculation
    p_joint = 0
    p_no_gene_trait = 1
    p_one_gene_trait = 1
    p_two_genes_trait = 1

    # iterate through people
    for person in people:
        print(f'person: {person}')
        p_gene = 0
        p_gene_mother = 0
        p_gene_mother_not = 0
        p_gene_father = 0
        p_gene_father_not = 0
        p_trait = 0
        parents = dict()
        parents = ancestors(people, person, parents)

        # calculate one_gene probability
        if person in one_gene:
            if parents:
                # conditional probability
                if len(parents) == 2:
                    # no grandparents
                    mother = parents["mother"]
                    father = parents["father"]
                    print(f'mother: {mother}')
                    print(f'father: {father}')

                    mother_gene_state = gene_state(mother, one_gene, two_genes)
                    father_gene_state = gene_state(father, one_gene, two_genes)
                    print(f'mother_gene_state: {mother_gene_state}')
                    print(f'father_gene_state: {father_gene_state}')

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

                    # either gene comes from mother, not father; or not from mother, but father
                    p_gene = p_gene_mother * p_gene_father_not + p_gene_mother_not * p_gene_father

                # add probabilities for 2 grandparents

                # add probabilities for 4 grandparents

            else:
                print('no parents')
                # no parents, unconditional probability
                p_gene = PROBS["gene"][1]

            # probability of trait / no trait with one gene
            if person in have_trait:
                p_trait = PROBS["trait"][1][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][1][False]

            # calculate probability of one gene and trait status
            p_one_gene_trait = p_gene * p_trait
            print(
                f'p_one_gene_trait: {p_one_gene_trait}')

        # calculate two_genes probability
        elif person in two_genes:
            if parents:
                # conditional probability
                print('two genes, has parents, calculate answer')

            else:
                print('no parents')
                # no parents, unconditional probability
                p_gene = PROBS["gene"][2]

            # probability of trait / no trait with two genes
            if person in have_trait:
                p_trait = PROBS["trait"][2][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][2][False]

            # calculate probability of two genes and trait status
            p_two_genes_trait = p_gene * p_trait
            print(f'p_two_genes_trait: {p_two_genes_trait}')

        # calculate no_gene probability
        else:
            if parents:
                # conditional probability
                print('no gene, has parents, calculate answer')
            else:
                print('no parents')
                # no parents, unconditional probability
                p_gene = PROBS["gene"][0]

            # probability of trait / no trait with no gene
            if person in have_trait:
                p_trait = PROBS["trait"][0][True]
            else:
                # probability of having no trait
                p_trait = PROBS["trait"][0][False]

            # calculate probability of no gene and trait status
            p_no_gene_trait = p_gene * p_trait
            print(
                f'p_no_gene_trait: {p_no_gene_trait}')

    p_joint = p_no_gene_trait * p_one_gene_trait * p_two_genes_trait
    print(f'p_joint probability: {p_joint}')

    print('leaving joint_probability function')
    return p_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
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
    print('in normalize')
    print(f'probabilities before normalization: {probabilities}')

    # get probabilites of gene and trait as lists for each person
    probabilities_copy = copy.deepcopy(probabilities)
    gene_list = []
    trait_list = []
    for person in probabilities_copy:
        print(f'gene content for {person}: {probabilities[person]["gene"]}')
        # append values to gene_list
        for i in range(3):
            gene_list.append(probabilities_copy[person]["gene"][i])

        # append values to trait_list
        trait_list.append(probabilities_copy[person]["trait"][True])
        trait_list.append(probabilities_copy[person]["trait"][False])

        print(f'gene_prob list: {gene_list}')
        print(f'trait_prob list: {trait_list}')

        # turn lists into numpy arrays
        gene_arr = np.asarray(gene_list)
        trait_arr = np.asarray(trait_list)

        # normalize arrays
        gene_arr = gene_arr / gene_arr.sum()
        trait_arr = trait_arr / trait_arr.sum()

        # turn normalized arrays back into lists
        gene_list = gene_arr.tolist()
        trait_list = trait_arr.tolist()

        # update probabilites with normalized values
        probabilities[person]["gene"][0] = gene_list[0]
        probabilities[person]["gene"][1] = gene_list[1]
        probabilities[person]["gene"][2] = gene_list[2]
        probabilities[person]["trait"][True] = trait_list[0]
        probabilities[person]["trait"][False] = trait_list[1]

    print(f'probabilities after normalization: {probabilities}')
    print('leaving normalize')


if __name__ == "__main__":
    main()
