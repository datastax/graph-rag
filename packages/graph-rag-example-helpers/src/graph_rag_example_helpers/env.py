import os
from enum import Enum, auto

from dotenv import find_dotenv, load_dotenv


class Environment(Enum):
    """Enumeration of supported environments for examples."""

    CASSIO = auto()
    """Environment variables for connecting to AstraDB via CassIO"""

    ASTRAPY = auto()
    """Environment variables for connecting to AstraDB via AstraPy"""

    def required_envvars(self) -> list[str]:
        """
        Return the required environment variables for this environment.

        Returns
        -------
        :
            The environment variables required in this environment.

        Raises
        ------
        ValueError
            If the environment isn't recognized.
        """
        required = ["OPENAI_API_KEY", "ASTRA_DB_APPLICATION_TOKEN"]
        if self == Environment.CASSIO:
            required.append("ASTRA_DB_DATABASE_ID")
        elif self == Environment.ASTRAPY:
            required.append("ASTRA_DB_API_ENDPOINT")
        else:
            raise ValueError(f"Unrecognized environment '{self}")
        return required


NON_SECRETS = {"ASTRA_DB_API_ENDPOINT", "ASTRA_DB_DATABASE_ID"}
"""Environment variables that can use `input` instead of `getpass`."""


def verify_environment(env: Environment = Environment.CASSIO):
    """Verify the necessary environment variables are set."""
    for required in env.required_envvars():
        assert required in os.environ, f'"{required}" not defined in environment'


def initialize_from_colab_userdata(env: Environment = Environment.CASSIO):
    """Try to initialize environment from colab `userdata`."""
    from google.colab import userdata  # type: ignore[import-untyped]

    for required in env.required_envvars():
        os.environ[required] = userdata.get(required)

    try:
        os.environ["ASTRA_DB_KEYSPACE"] = userdata.get("ASTRA_DB_KEYSPACE")
    except userdata.SecretNotFoundError as _:
        # User doesn't have a keyspace set, so use the default.
        os.environ.pop("ASTRA_DB_KEYSPACE", None)

    try:
        os.environ["LANGCHAIN_API_KEY"] = userdata.get("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "True"
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        print("Colab Secret not set / accessible. Not configuring tracing")  # noqa: T201
        os.environ.pop("LANGCHAIN_API_KEY")
        os.environ.pop("LANGCHAIN_TRACING_V2")


def initialize_from_prompts(env: Environment = Environment.CASSIO):
    """Initialize the environment by prompting the user."""
    import getpass

    for required in env.required_envvars():
        if required in os.environ:
            continue
        elif required in NON_SECRETS:
            os.environ[required] = input(required)
        else:
            os.environ[required] = getpass.getpass(required)


def initialize_environment(env: Environment = Environment.CASSIO):
    """
    Initialize the environment variables.

    Parameters
    ----------
    env :
        The environment to initialize

    Notes
    -----
        This uses the following:

        1. If a `.env` file is found, load environment variables from that.
        2. If not, and running in colab, set necessary environment variables from
            secrets.
        3. If necessary variables aren't set by the above, then prompts the user.
    """
    # 1. If a `.env` file is found, load environment variables from that.
    if dotenv_path := find_dotenv():
        load_dotenv(dotenv_path)
        verify_environment(env)
        return

    # 2. If not, and running in colab, set necesary environment variables from secrets.
    try:
        initialize_from_colab_userdata(env)
        verify_environment(env)
        return
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Initialize from prompts.
    initialize_from_prompts(env)
    verify_environment(env)
