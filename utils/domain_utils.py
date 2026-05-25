HUMAN_PREFERENCE_DOMAINS = {
    "search_arena",
    "paper_review",
    "paper_writer_review",
    "code_review",
}
HUMAN_PREFERENCE_AND_GRADING_DOMAINS = {*HUMAN_PREFERENCE_DOMAINS, "imo_grading"}
EVALUATOR_DEPENDENT_PRODUCER_DOMAINS = {"paper_writer_review", "imo_proof"}
ACCURACY_REPORT_DOMAINS = {*HUMAN_PREFERENCE_AND_GRADING_DOMAINS, "imo_proof"}


def get_domain_score_key(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return "overall_accuracy"
    # Balrog game domains
    elif "balrog" in domain:
        return "average_progress"
    # Genesis robotic control domains
    elif "genesis" in domain:
        return "average_fitness"
    # Polyglot domain
    elif "polyglot" in domain:
        return "accuracy_score"


def get_domain_splits(domain, eval_test=False):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        splits = ["train", "val"]
        if eval_test:
            splits.append("test")
        return splits
    # Balrog game domains
    elif "balrog" in domain:
        return ["train"]
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ["train"]
    # Polyglot domain
    elif "polyglot" in domain:
        return ["train"]


def can_domain_ensembled(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return True
    # Balrog game domains
    elif "balrog" in domain:
        return False
    # Genesis robotic control domains
    elif "genesis" in domain:
        return False
    # Polyglot domain
    elif "polyglot" in domain:
        return False


def get_domain_eval_subset(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return "_filtered_100_train"
    # Balrog game domains
    elif "balrog" in domain:
        return ""
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ""
    # Polyglot domain
    elif "polyglot" in domain:
        return ""


def get_domain_test_subset(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return "_filtered_100_test"
    # Balrog game domains
    elif "balrog" in domain:
        return ""
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ""
    # Polyglot domain
    elif "polyglot" in domain:
        return ""


def get_domain_stagedeval_samples(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return 10
    # Balrog game domains
    elif "balrog" in domain:
        return 1
    # Genesis robotic control domains
    elif "genesis" in domain:
        return 3
    # Polyglot domain
    elif "polyglot" in domain:
        return 10


def get_domain_stagedeval_frac(domain):
    # NOTE: this is hardcoded wrt get_domain_stagedeval_samples and default domain configs
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return 10/100
    # Balrog game domains
    elif "balrog_babyai" in domain:
        return 1/10
    elif "balrog_minihack" in domain:
        return 1/5
    # Genesis robotic control domains
    elif "genesis" in domain:
        return 3/6
    # Polyglot domain
    elif "polyglot" in domain:
        return 10/60


def has_domain_val_subset(domain):
    # Human preferences domains
    if domain in ACCURACY_REPORT_DOMAINS:
        return True
    # Balrog game domains
    elif "balrog" in domain:
        return False
    # Genesis robotic control domains
    elif "genesis" in domain:
        return False
    # Polyglot domain
    elif "polyglot" in domain:
        return False
