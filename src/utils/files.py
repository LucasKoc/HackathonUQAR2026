import os
import shutil

from config import Config


class Files:
    @staticmethod
    def migration_p2():
        for split in [Config.PATH_TRAIN, Config.PATH_TEST]:
            src = Config.DATASET_PATH_P1 + split
            dst = Config.DATASET_PATH_P2 + split

            species_folders = [
                f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))
            ]

            for species in species_folders:
                # Check if species from P1 are already in P2
                if os.path.exists(os.path.join(dst, species)):
                  continue

                src_species = os.path.join(src, species)
                dst_species = os.path.join(dst, species)

                print(f"Moving {species} from {split} P1 to P2...")

                # copytree copies the entire directory and its contents
                shutil.copytree(src_species, dst_species, dirs_exist_ok=True)
