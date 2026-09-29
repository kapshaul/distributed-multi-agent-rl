import configparser



CONFIG_PATH_PPO = "./framework/ppo/config.ini"

def main(algorithm="PPO"):
    # Load configuration
    config = configparser.ConfigParser()
    if algorithm == "PPO":
        from framework import ppo

        config.read(CONFIG_PATH_PPO)
        ppo(config=config)


if __name__ == "__main__":
    main(algorithm="PPO")
    