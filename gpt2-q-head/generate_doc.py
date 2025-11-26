#!/usr/bin/env python3
"""Generate a PDF document explaining the GPT2 Q-head objective function."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor


def create_pdf():
    doc = SimpleDocTemplate(
        "gpt2_q_head_overview.pdf",
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=HexColor('#2c3e50')
    )

    subsection_style = ParagraphStyle(
        'Subsection',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
        textColor=HexColor('#34495e')
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceBefore=4,
        spaceAfter=8,
        leading=14
    )

    equation_style = ParagraphStyle(
        'Equation',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Courier',
        leftIndent=30,
        spaceBefore=8,
        spaceAfter=8,
        backColor=HexColor('#f5f5f5')
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Courier',
        leftIndent=20,
        spaceBefore=6,
        spaceAfter=6,
        backColor=HexColor('#f0f0f0')
    )

    story = []

    # Title
    story.append(Paragraph("GPT-2 with Q-Head: Technical Overview", title_style))
    story.append(Spacer(1, 12))

    # Abstract
    story.append(Paragraph("1. Abstract", section_style))
    story.append(Paragraph(
        "This document describes a method for augmenting GPT-2 with a Q-head that predicts "
        "expected future returns for each possible next token. The approach combines standard "
        "language modeling with reinforcement learning concepts, enabling the model to learn "
        "both token prediction and value estimation simultaneously.",
        body_style
    ))

    # Main Idea
    story.append(Paragraph("2. Main Idea", section_style))
    story.append(Paragraph(
        "The core insight is to treat autoregressive language generation as a sequential "
        "decision-making process. At each position t in a sequence, the model must choose "
        "the next token a from the vocabulary V. We augment GPT-2 with a second output head "
        "that predicts Q(s<sub>t</sub>, a) — the expected discounted future return if action a is taken "
        "in state s<sub>t</sub>.",
        body_style
    ))

    story.append(Paragraph("Architecture Overview:", subsection_style))
    story.append(Paragraph(
        "<b>• Backbone:</b> GPT-2 transformer producing hidden states h<sub>t</sub> of dimension H<br/>"
        "<b>• LM Head:</b> Linear projection h<sub>t</sub> → logits over vocabulary (standard next-token prediction)<br/>"
        "<b>• Q Head:</b> Linear projection h<sub>t</sub> → Q-values over vocabulary (value prediction for each action)",
        body_style
    ))
    story.append(Paragraph(
        "The Q-head outputs a |V|-dimensional vector at each position, where each component "
        "Q(s<sub>t</sub>, a) represents the expected future return if token a is selected as the next token. "
        "This allows querying the value of any action without needing to actually sample it.",
        body_style
    ))

    # Objective Function
    story.append(Paragraph("3. Objective Function", section_style))
    story.append(Paragraph(
        "The training objective combines two loss terms:",
        body_style
    ))
    story.append(Paragraph("L<sub>total</sub> = L<sub>LM</sub> + λ · L<sub>Q</sub>", equation_style))
    story.append(Paragraph(
        "where λ (q_weight) controls the relative importance of the Q-learning objective.",
        body_style
    ))

    # LM Loss
    story.append(Paragraph("3.1 Language Modeling Loss (L<sub>LM</sub>)", subsection_style))
    story.append(Paragraph(
        "Standard causal language modeling loss using cross-entropy. For a sequence of tokens "
        "(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>), the model predicts each token given all previous tokens:",
        body_style
    ))
    story.append(Paragraph(
        "L<sub>LM</sub> = −Σ<sub>t=1</sub><sup>T−1</sup> log P(x<sub>t+1</sub> | x<sub>1</sub>, ..., x<sub>t</sub>)",
        equation_style
    ))
    story.append(Paragraph(
        "This is computed using shifted logits: predictions at positions 0..T−2 are compared "
        "against ground truth tokens at positions 1..T−1. Padding positions are masked out.",
        body_style
    ))

    # Q Loss
    story.append(Paragraph("3.2 Q-Value Loss (L<sub>Q</sub>)", subsection_style))
    story.append(Paragraph(
        "The Q-head is trained using mean squared error between predicted Q-values and "
        "discounted return targets. For the observed action a<sub>t</sub> (the actual next token), "
        "we supervise:",
        body_style
    ))
    story.append(Paragraph(
        "L<sub>Q</sub> = (1/N) · Σ<sub>t</sub> (Q(s<sub>t</sub>, a<sub>t</sub>) − G<sub>t</sub>)²",
        equation_style
    ))
    story.append(Paragraph("where G<sub>t</sub> is the discounted return from position t+1 onward, with bootstrapping:", body_style))
    story.append(Paragraph(
        "G<sub>t</sub> = Σ<sub>k=t+1</sub><sup>L−1</sup> γ<sup>k−(t+1)</sup> · r<sub>k</sub> + γ<sup>L−1−t</sup> · V<sub>bootstrap</sub>",
        equation_style
    ))
    story.append(Paragraph(
        "Here γ is the discount factor (default 0.99) and r<sub>k</sub> is the reward at position k. "
        "The bootstrap term V<sub>bootstrap</sub> handles the context window boundary (see Section 4).",
        body_style
    ))

    # Computing Discounted Returns
    story.append(Paragraph("4. Computing Discounted Returns with Bootstrapping", section_style))
    story.append(Paragraph(
        "A key challenge is handling the context window boundary. The sequence doesn't truly end "
        "at position L—it's an arbitrary truncation. Treating it as a terminal state (future value = 0) "
        "would bias Q-estimates downward near the boundary.",
        body_style
    ))

    story.append(Paragraph("4.1 Bootstrap Value at Context Boundary", subsection_style))
    story.append(Paragraph(
        "Instead, we bootstrap using the expected Q-value under the current policy at the last position:",
        body_style
    ))
    story.append(Paragraph(
        "V<sub>bootstrap</sub> = E<sub>a∼π</sub>[Q(s<sub>L−1</sub>, a)] = Σ<sub>a</sub> π(a|s<sub>L−1</sub>) · Q(s<sub>L−1</sub>, a)",
        equation_style
    ))
    story.append(Paragraph(
        "where π(a|s) = softmax(logits) is the policy distribution from the LM head. This computes "
        "the inner product between the predicted next-token distribution and the Q-values over all actions, "
        "giving an estimate of expected future value beyond the context window.",
        body_style
    ))

    story.append(Paragraph("4.2 Backward Dynamic Programming", subsection_style))
    story.append(Paragraph(
        "The discounted returns are computed via dynamic programming, working backwards "
        "from the bootstrap value:",
        body_style
    ))

    code_text = """def compute_discounted_returns(rewards, gamma, bootstrap):
    returns = zeros_like(rewards)
    future = bootstrap  # E_π[Q(s_L, a)] instead of 0
    for t in range(L-1, -1, -1):
        returns[t] = future
        future = rewards[t] + gamma * future
    return returns"""
    story.append(Preformatted(code_text, code_style))

    story.append(Paragraph(
        "This ensures that returns[t] properly accounts for value beyond the context window. "
        "For n tokens remaining until the boundary, this effectively computes an n-step TD target "
        "that bootstraps with the expected Q-value.",
        body_style
    ))

    # Key Properties
    story.append(Paragraph("5. Key Properties", section_style))

    story.append(Paragraph(
        "<b>1. On-Policy Supervision:</b> The Q-head is only supervised for the actually observed "
        "next token (the action that was taken). This is a form of on-policy learning where "
        "we learn Q-values along the trajectory defined by the training data.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Vocabulary-Sized Q-Output:</b> Unlike scalar value functions V(s), the Q-head outputs "
        "a value for every possible action. This enables action selection via argmax<sub>a</sub> Q(s, a) "
        "at inference time.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Shared Representations:</b> Both heads share the transformer backbone, allowing the "
        "Q-head to benefit from the rich representations learned for language modeling.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4. Flexible Reward Specification:</b> Rewards can encode any per-token signal — correctness, "
        "style, safety scores, or learned reward models. The current implementation uses "
        "synthetic random rewards for demonstration.",
        body_style
    ))

    # Extensions
    story.append(Paragraph("6. Extensions and Future Directions", section_style))
    story.append(Paragraph(
        "The current implementation provides on-policy Q-learning. For more sophisticated "
        "applications, consider:",
        body_style
    ))
    story.append(Paragraph(
        "<b>• Off-policy learning:</b> Using importance sampling or model-based rollouts to learn "
        "Q-values for actions not taken in the training data.<br/><br/>"
        "<b>• Temporal difference learning:</b> Using TD(0) or TD(λ) updates instead of "
        "Monte Carlo returns.<br/><br/>"
        "<b>• Actor-critic methods:</b> Using the Q-head to guide policy updates for the LM head.<br/><br/>"
        "<b>• Reward modeling:</b> Training a separate reward model and using its outputs as r<sub>t</sub>.",
        body_style
    ))

    # Summary
    story.append(Paragraph("7. Summary", section_style))
    story.append(Paragraph(
        "The GPT2-with-Q-head architecture demonstrates how reinforcement learning concepts "
        "can be integrated into language models. The combined objective:",
        body_style
    ))
    story.append(Paragraph("L = L<sub>LM</sub> + λ · L<sub>Q</sub>", equation_style))
    story.append(Paragraph(
        "trains the model to simultaneously predict next tokens (via cross-entropy) and "
        "estimate expected future returns (via MSE on discounted returns). This provides "
        "a foundation for value-guided text generation and reinforcement learning from "
        "human or automated feedback.",
        body_style
    ))

    doc.build(story)
    print("PDF generated: gpt2_q_head_overview.pdf")


if __name__ == "__main__":
    create_pdf()
