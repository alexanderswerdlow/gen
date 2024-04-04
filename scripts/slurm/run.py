import slurminade

slurminade.update_default_configuration(partition="kate_reserved", exclusive=False)

@slurminade.node_setup
def setup():
    print("I will run automatically on every slurm node at the beginning!")


@slurminade.slurmify()
def clean_up():
    print("Clean up")


if __name__ == "__main__":
    slurminade.join()  # make sure that no job runs before prepare has finished
    clean_up.distribute()