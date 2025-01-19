import globals from "globals";
import pluginJs from "@eslint/js";

/** @type {import('eslint').Linter.Config[]} */
export default [
  {
    languageOptions: {
      globals: {
        ...globals.es5,
        ...globals.worker,
        ort: "readonly",
      }
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": "warn",
      "no-use-before-define": "error",
      "semi": ["error", "always"],
    },
  },
  ...pluginJs.configs.recommended,
];