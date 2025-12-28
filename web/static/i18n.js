// Internationalization for Web App

const translations = {
    ja: {
        app_title: "PyTorch Tensor Marathon",
        welcome: "テンソル操作の練習へようこそ！",
        welcome_subtitle: "100問の厳選された問題でPyTorchのテンソル操作をマスターしましょう",
        categories: "カテゴリ",
        total_problems: "総問題数",
        categories_count: "カテゴリ数",
        your_progress: "進捗率",
        select_category_instruction: "サイドバーからカテゴリを選択して開始してください",

        // Instructions
        how_to_use: "📚 使い方",
        instruction_1: "サイドバーからカテゴリを選択",
        instruction_2: "リストから問題を選択",
        instruction_3: "解答コードを記述",
        instruction_4: "「実行」をクリックして答え合わせ",
        important_rules: "⚠️ 重要なルール",
        rule_1: "結果は必ず <code>result</code> 変数に代入してください",
        rule_2: "コードは形状と値の両方がチェックされます",
        rule_3: "困ったら「ヒント」ボタンを使いましょう",
        rule_4: "進捗はブラウザに自動保存されます",

        // Categories
        cat_reshape_permute: "🔄 Reshape & Permute",
        cat_indexing_slicing: "🎯 Indexing & Slicing",
        cat_broadcasting: "📡 Broadcasting",
        cat_gather_scatter: "🎲 Gather & Scatter",
        cat_einsum: "∑ Einstein Summation",
        cat_stacking_splitting: "📚 Stacking & Splitting",
        cat_advanced_ops: "⚡ Advanced Operations",
        cat_dl_applications: "🧠 DL Applications",

        // Problem view
        back: "リストに戻る",
        setup_code: "📝 セットアップコード",
        your_solution: "💻 あなたの解答",
        hint: "ヒント",
        run: "実行",
        show_solution: "解答を表示",
        expected_solution: "✅ 期待される解答",
        previous: "← 前の問題",
        next: "次の問題 →",

        // Results
        correct_title: "✅ 正解！",
        correct_message: "形状と値が一致しています！",
        incorrect_title: "❌ 不正解",
        shape_error: "形状エラー",
        value_error: "値エラー",
        execution_error: "実行エラー",
        expected_shape: "期待される形状",
        actual_shape: "実際の形状",

        // Difficulty
        beginner: "初級",
        intermediate: "中級",
        advanced: "上級",
        expert: "エキスパート",

        // Gemini AI Features
        ai_explain: "AI解説",
        ai_hint: "AIヒント",

    },

    en: {
        app_title: "PyTorch Tensor Marathon",
        welcome: "Welcome to Tensor Operation Practice!",
        welcome_subtitle: "Master PyTorch tensor operations with 100 curated problems",
        categories: "Categories",
        total_problems: "Total Problems",
        categories_count: "Categories",
        your_progress: "Your Progress",
        select_category_instruction: "Select a category from the sidebar to begin",

        // Instructions
        how_to_use: "📚 How to Use",
        instruction_1: "Select a category from the sidebar",
        instruction_2: "Choose a problem from the list",
        instruction_3: "Write your solution code",
        instruction_4: "Click \"Run\" to check your answer",
        important_rules: "⚠️ Important Rules",
        rule_1: "Always assign your result to the variable <code>result</code>",
        rule_2: "Your code will be checked for both shape and values",
        rule_3: "Use the \"Hint\" button if you're stuck",
        rule_4: "Progress is saved automatically in your browser",

        // Categories
        cat_reshape_permute: "🔄 Reshape & Permute",
        cat_indexing_slicing: "🎯 Indexing & Slicing",
        cat_broadcasting: "📡 Broadcasting",
        cat_gather_scatter: "🎲 Gather & Scatter",
        cat_einsum: "∑ Einstein Summation",
        cat_stacking_splitting: "📚 Stacking & Splitting",
        cat_advanced_ops: "⚡ Advanced Operations",
        cat_dl_applications: "🧠 DL Applications",

        // Problem view
        back: "Back to List",
        setup_code: "📝 Setup Code",
        your_solution: "💻 Your Solution",
        hint: "Hint",
        run: "Run",
        show_solution: "Show Solution",
        expected_solution: "✅ Expected Solution",
        previous: "← Previous",
        next: "Next →",

        // Results
        correct_title: "✅ Correct!",
        correct_message: "Shape and values match!",
        incorrect_title: "❌ Incorrect",
        shape_error: "Shape Error",
        value_error: "Value Error",
        execution_error: "Execution Error",
        expected_shape: "Expected Shape",
        actual_shape: "Actual Shape",

        // Difficulty
        beginner: "Beginner",
        intermediate: "Intermediate",
        advanced: "Advanced",
        expert: "Expert",

        // Gemini AI Features
        ai_explain: "AI Explanation",
        ai_hint: "AI Hint",

    }
};

function t(key, lang = 'en') {
    return translations[lang][key] || key;
}
