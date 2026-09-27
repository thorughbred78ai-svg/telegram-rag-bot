from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from .config import settings
from .telegram_bot import (
    answer,
    clear_command,
    help_command,
    start,
)


def main() -> None:

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "clear",
            clear_command,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            answer,
        )
    )

    application.run_polling(
        allowed_updates=["message"],
    )


if __name__ == "__main__":
    main()
