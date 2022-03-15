import copy
import csv
import itertools
import sys

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


def mothers(people, person, ancestor_list):
    # recursively get mothers
    new_mother = people.get(person, {}).get("mother", None)
    if new_mother == None:
        # base case
        return ancestor_list
    ancestor_list.append(new_mother)
    return mothers(people, new_mother, ancestor_list)


def fathers(people, person, ancestor_list):
    # recursively get fathers
    new_father = people.get(person, {}).get("father", None)
    if new_father == None:
        # base case
        return ancestor_list
    ancestor_list.append(new_father)
    return fathers(people, new_father, ancestor_list)


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
    one_gene = {"Harry"}
    two_genes = {"James"}
    has_trait = {"James"}

    print(f'people: {people}')
    print(f'one_gene: {one_gene}')
    print(f'two_genes: {two_genes}')
    print(f'have_trait: {have_trait}')

    # unconditional one_gene probabilities
    u_one_gene_list = list(one_gene)
    u_p_one_gene = []
    if len(u_one_gene_list) != 0:
        for person in u_one_gene_list:
            u_p_one_gene.append(PROBS["gene"][1])

    # unconditional two_genes probabilities
    u_two_genes_list = list(two_genes)
    u_p_two_genes = []
    if len(u_two_genes_list) != 0:
        for person in u_two_genes_list:
            u_p_two_genes.append(PROBS["gene"][2])

    # unconditional no_gene probabilities
    u_no_gene_list = []
    u_p_no_gene = []
    for person in people:
        if person not in one_gene and person not in two_genes:
            u_no_gene_list.append(person)

    if len(u_no_gene_list) != 0:
        for person in u_no_gene_list:
            u_p_no_gene.append(PROBS["gene"][0])

    # have_trait probabilities
    have_trait_list = list(have_trait)
    p_have_trait = []
    if len(have_trait_list) != 0:
        for person in have_trait_list:
            p_have_trait.append(PROBS["trait"][0][True])

    # no_trait probabilities, given no gene
    no_trait_no_gene_list = []
    p_no_trait_no_gene = []

    for person in people:
        if person not in have_trait:
            if person not in one_gene:
                if person not in two_genes:
                    no_trait_no_gene_list.append(person)

    if len(no_trait_no_gene_list) != 0:
        for person in no_trait_no_gene_list:
            p_no_trait_no_gene.append(PROBS["trait"][0][False])

    print(f'u_one_gene_list: {u_one_gene_list}')
    print(f'u_p_one_gene: {u_p_one_gene}')
    print(f'u_two_genes_list: {u_two_genes_list}')
    print(f'u_p_two_genes: {u_p_two_genes}')
    print(f'u_no_gene_list: {u_no_gene_list}')
    print(f'u_p_no_gene: {u_p_no_gene}')
    print(f'have_trait_list: {have_trait_list}')
    print(f'p_have_trait: {p_have_trait}')
    print(f'no_trait_no_gene_list: {no_trait_no_gene_list}')
    print(f'p_no_trait_no_gene: {p_no_trait_no_gene}')

    # iterate through people
    for person in people:
        p = 0
        p_trait = 0

        have_parents = False
        if people[person]["mother"] or people[person]["father"]:
            have_parents = True

        # calculate one_gene probability
        if person in one_gene:
            if not have_parents:
                # no parents, use unconditional probability
                p = PROBS["trait"][1]
            else:
                # have parents, calculate probability
                print('getting parents')
                mothers_list = []
                mothers_list = mothers(people, person, mothers_list)
                fathers_list = []
                fathers_list = fathers(people, person, fathers_list)
                print('back in joint_probability')
                print(f'mothers_list: {mothers_list}')
                print(f'fathers_list: {fathers_list}')
                # gets gene from mother, not father

                # gets gene from father, not mother

            if person in have_trait:
                p_trait = PROBS["trait"][1][True]
            else:
                p_trait = PROBS["trait"][1][False]

        # calculate two_genes probability
        if person in two_genes:
            if not have_parents:
                p = PROBS["gene"][2]

            if person in have_trait:
                p_trait = PROBS["trait"][2][True]
            else:
                p_trait = PROBS["trait"][2][False]

            p_two_genes = p * p_trait
            print(f'p_two_genes: {p_two_genes}')


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
