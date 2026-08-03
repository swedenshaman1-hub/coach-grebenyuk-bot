import json
import unittest

from strict_contract import (
    ResultStatus,
    SourceInfo,
    build_strict_prompt,
    parse_and_validate,
    render_verified_answer,
)


SOURCES = [SourceInfo(id="src-1", title="Метод пяти единичек")]


def payload(**overrides):
    value = {
        "status": "verified",
        "answer": "Метод требует выбрать одну целевую аудиторию. Что выберешь сегодня?",
        "claims": [
            {
                "text": "Метод требует выбрать одну целевую аудиторию",
                "evidence": "Нужно выбрать одну целевую аудиторию и не распыляться",
                "source": "Метод пяти единичек",
                "citation": "[1]",
            }
        ],
        "missing_information": [],
        "confidence": "high",
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


class StrictContractTests(unittest.TestCase):
    def test_verified_answer_requires_real_evidence(self):
        result = parse_and_validate(payload(), SOURCES)
        self.assertTrue(result.is_verified)
        self.assertEqual(result.claims[0].source_id, "src-1")

    def test_partial_answer_drops_unknown_source(self):
        raw = json.loads(payload())
        raw["claims"][0]["source"] = "Несуществующий источник"
        result = parse_and_validate(json.dumps(raw, ensure_ascii=False), SOURCES)
        self.assertEqual(result.status, ResultStatus.PARTIAL)
        self.assertFalse(result.answer)

    def test_insufficient_never_returns_substantive_answer(self):
        result = parse_and_validate(
            payload(
                status="insufficient",
                answer="Правдоподобная догадка",
                claims=[],
                missing_information=["Нет сведений о рынке"],
                confidence="insufficient",
            ),
            SOURCES,
        )
        self.assertEqual(result.status, ResultStatus.INSUFFICIENT)
        self.assertEqual(result.answer, "")

    def test_renderer_removes_an_uncovered_new_fact(self):
        result = parse_and_validate(
            payload(
                answer=(
                    "Метод требует выбрать одну целевую аудиторию. "
                    "Это гарантированно увеличит прибыль в десять раз."
                )
            ),
            SOURCES,
        )
        self.assertTrue(result.is_verified)
        rendered = render_verified_answer(result)
        self.assertNotIn("гарантированно", rendered)
        self.assertIn("одну целевую аудиторию", rendered)

    def test_raw_response_keeps_citation_markers(self):
        result = parse_and_validate(payload(), SOURCES)
        self.assertIn("[1]", result.raw_answer)
        self.assertEqual(result.claims[0].citation, "[1]")

    def test_russian_grammatical_forms_keep_valid_evidence(self):
        result = parse_and_validate(
            payload(
                answer="Метод включает выбор одной боли аудитории.",
                claims=[{
                    "text": "Метод включает выбор одной боли аудитории",
                    "evidence": "выбрать только одну боль для этой аудитории",
                    "source": "Метод пяти единичек",
                    "citation": "[1]",
                }],
            ),
            SOURCES,
        )
        self.assertTrue(result.is_verified)

    def test_prompt_forbids_external_knowledge(self):
        prompt = build_strict_prompt("Как расти?", [])
        self.assertIn("Запрещено использовать интернет", prompt)
        self.assertIn("фоновые знания модели", prompt)


if __name__ == "__main__":
    unittest.main()
