"""Command-line interface for the Tensor Marathon."""

import sys
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box

from torch_tensor_marathon.config import config
from torch_tensor_marathon.problem import Problem, problem_bank
from torch_tensor_marathon.problems import initialize_problems
from torch_tensor_marathon.checker import CorrectnessChecker
from torch_tensor_marathon.i18n import t, get_category_name

console = Console()


class TensorMarathonCLI:
    """Interactive CLI for the Tensor Marathon."""

    def __init__(self):
        self.checker = CorrectnessChecker()
        initialize_problems()

    def run(self):
        """Main entry point for the CLI."""
        self._display_welcome()

        while True:
            try:
                self._show_main_menu()
            except KeyboardInterrupt:
                console.print("\n\n👋 さようなら！ / Goodbye!")
                break
            except EOFError:
                break

    def _display_welcome(self):
        """Display welcome message."""
        lang = config.language
        title = t("app_title", lang)
        welcome = t("welcome", lang)

        welcome_text = f"""
{title}

{welcome}

問題数 / Total Problems: **{len(problem_bank)}**
カテゴリ数 / Categories: **{len(problem_bank.get_categories())}**
"""
        console.print(Panel(Markdown(welcome_text), border_style="cyan", box=box.DOUBLE))

    def _show_main_menu(self):
        """Show main menu and handle selection."""
        lang = config.language

        console.print(f"\n[bold cyan]{t('select_category', lang)}[/bold cyan]")

        # Display categories
        categories = problem_bank.get_categories()

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column(t("category", lang), style="cyan")
        table.add_column("Problems", justify="right", style="green")

        for idx, cat in enumerate(categories, 1):
            stats = problem_bank.get_category_stats(cat)
            cat_name = get_category_name(cat, lang)
            table.add_row(str(idx), cat_name, str(stats["total"]))

        console.print(table)
        console.print("\n[dim]0. 言語切り替え / Change Language[/dim]")
        console.print("[dim]q. 終了 / Quit[/dim]\n")

        choice = Prompt.ask("選択 / Select", default="1")

        if choice.lower() == 'q':
            raise KeyboardInterrupt
        elif choice == '0':
            self._change_language()
            return

        try:
            cat_idx = int(choice) - 1
            if 0 <= cat_idx < len(categories):
                self._show_category(categories[cat_idx])
            else:
                console.print("[red]無効な選択です / Invalid selection[/red]")
        except ValueError:
            console.print("[red]無効な入力です / Invalid input[/red]")

    def _change_language(self):
        """Change UI language."""
        current = config.language
        new_lang = "en" if current == "ja" else "ja"
        config.set_language(new_lang)
        console.print(f"[green]✓ Language changed to: {new_lang.upper()}[/green]")

    def _show_category(self, category: str):
        """Show problems in a category."""
        lang = config.language
        problems = problem_bank.get_problems_by_category(category)

        console.print(f"\n[bold magenta]{get_category_name(category, lang)}[/bold magenta]\n")

        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column(t("problem", lang), style="cyan", no_wrap=False)
        table.add_column(t("difficulty", lang), justify="center")

        for idx, prob in enumerate(problems, 1):
            title = prob.get_title(lang)
            difficulty = t(prob.difficulty, lang)
            table.add_row(str(idx), title, difficulty)

        console.print(table)
        console.print("\n[dim]0. 戻る / Back[/dim]\n")

        choice = Prompt.ask("問題を選択 / Select problem", default="0")

        if choice == '0':
            return

        try:
            prob_idx = int(choice) - 1
            if 0 <= prob_idx < len(problems):
                self._practice_problem(problems[prob_idx])
            else:
                console.print("[red]無効な選択です / Invalid selection[/red]")
        except ValueError:
            console.print("[red]無効な入力です / Invalid input[/red]")

    def _practice_problem(self, problem: Problem):
        """Practice a single problem."""
        lang = config.language

        # Display problem
        console.print(f"\n[bold green]━━━ {problem.get_title(lang)} ━━━[/bold green]\n")

        # Description
        desc_panel = Panel(
            problem.get_description(lang),
            title=f"📝 {t('goal', lang)}",
            border_style="blue"
        )
        console.print(desc_panel)

        # Show hint if available
        hint = problem.get_hint(lang)
        if hint:
            if Confirm.ask(f"\n💡 {t('hint', lang)}を表示しますか？ / Show hint?", default=False):
                console.print(Panel(hint, title=t("hint", lang), border_style="yellow"))

        # Show setup code
        console.print(f"\n[bold cyan]Setup Code:[/bold cyan]")
        syntax = Syntax(problem.get_full_setup_code(), "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Template
        console.print(f"\n[bold cyan]Your Task:[/bold cyan]")
        console.print("[dim]# Write your solution below:[/dim]")
        console.print("[yellow]# result = ...[/yellow]\n")

        # Get user input
        console.print("[bold]💻 解答を入力してください / Enter your solution:[/bold]")
        console.print("[dim](複数行対応: 空行を2回入力で終了 / Multi-line: press Enter twice to finish)[/dim]\n")

        user_lines = []
        empty_count = 0

        while empty_count < 2:
            try:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                else:
                    empty_count = 0
                user_lines.append(line)
            except EOFError:
                break

        user_code = "\n".join(user_lines).strip()

        if not user_code:
            console.print("[yellow]スキップしました / Skipped[/yellow]")
            return

        # Check solution
        console.print("\n[bold]🔍 チェック中 / Checking...[/bold]")

        result = self.checker.check_solution(
            setup_code=problem.setup_code,
            user_code=user_code,
            solution_code=problem.solution_code,
            expected_shape=problem.expected_shape,
        )

        # Display result
        if result.is_correct:
            console.print(Panel(
                f"[bold green]{t('correct', lang)}[/bold green]\n\n"
                f"{t('result_shape', lang)}: {result.actual_shape}",
                title="✅ Success",
                border_style="green",
                box=box.DOUBLE
            ))
        else:
            error_msg = result.message
            if result.error_type == "shape":
                error_msg = t("shape_error", lang,
                             expected=result.expected_shape,
                             actual=result.actual_shape)
            elif result.error_type == "value":
                error_msg = t("value_error", lang)

            console.print(Panel(
                f"[bold red]{t('incorrect', lang)}[/bold red]\n\n"
                f"{error_msg}",
                title="❌ Failed",
                border_style="red"
            ))

        # Show solution
        if Confirm.ask("\n📖 解答を表示しますか？ / Show solution?", default=False):
            console.print("\n[bold cyan]Expected Solution:[/bold cyan]")
            solution_syntax = Syntax(problem.solution_code, "python", theme="monokai", line_numbers=True)
            console.print(solution_syntax)

        input("\n[dim]Press Enter to continue...[/dim]")
