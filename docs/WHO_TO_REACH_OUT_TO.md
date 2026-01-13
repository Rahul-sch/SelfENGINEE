# Who To Reach Out To

## How To Use This Document

Each person is categorized by:
- **Tier**: How likely they are to respond/care
- **Why**: Specific reason they'd be interested
- **Pitch**: Exact 2-sentence message to send
- **Contact**: How to reach them

---

# TIER 1: Most Likely to Care

## AI Safety Researchers

### Jan Leike
- **Role:** Former Head of Alignment, OpenAI (now Anthropic)
- **Why:** Leading AI safety researcher, cares about secure AI outputs
- **Pitch:** "I built SABS - a novel algorithm that integrates security verification into LLM beam search during code generation. The paper + working implementation is on GitHub with 158 tests passing."
- **Contact:** Twitter @janleike, LinkedIn

### Dario Amodei
- **Role:** CEO, Anthropic
- **Why:** Anthropic's entire mission is AI safety
- **Pitch:** "Built 3 tools for AI code safety: a prompt refinement system, code scanner for AI hallucinations, and a novel beam search algorithm that prunes insecure paths during generation. All shipped and tested."
- **Contact:** Through Anthropic contact, Twitter @DarioAmodei

### Chris Olah
- **Role:** Co-founder, Anthropic (Research)
- **Why:** Deep interest in interpretability and safety
- **Pitch:** "SABS (Security-Aware Beam Search) integrates AST analysis into the decoding loop - similar philosophy to your interpretability work but for code security."
- **Contact:** Twitter @ch402

---

## Developer Tools VCs

### Guillermo Rauch
- **Role:** CEO, Vercel
- **Why:** Invests in developer tools, Next.js ecosystem
- **Pitch:** "Built VibeGuard - antivirus for AI-generated code. Catches hallucinated packages and hardcoded secrets before they ship. 14K lines, ships as npm CLI."
- **Contact:** Twitter @rauchg, Email via Vercel

### Nat Friedman
- **Role:** Former CEO GitHub, Active Angel
- **Why:** Invested in Copilot, cares about AI dev tools
- **Pitch:** "VibeGuard scans AI-generated code for the problems Copilot creates - hallucinated dependencies, leaked secrets, insecure defaults. Auto-fixes with AI explanations."
- **Contact:** Twitter @natfriedman

### Patrick Collison
- **Role:** CEO, Stripe
- **Why:** Known to respond to interesting projects, technical founder
- **Pitch:** "Built 3 AI safety tools in 2 weeks: prompt refinement SaaS, AI code scanner, and a novel algorithm for secure code generation. All shipped with production-quality code."
- **Contact:** Twitter @patrickc, Email patrick@stripe.com

---

## Security Leaders

### Troy Hunt
- **Role:** Founder, Have I Been Pwned
- **Why:** Developer security advocate, massive audience
- **Pitch:** "VibeGuard catches hardcoded secrets and AI hallucinations before they hit production. Think of it as pre-commit security for the AI coding era."
- **Contact:** Twitter @troaborhunt, troyhunt.com contact

### Haroon Meer
- **Role:** Founder, Thinkst (Canary)
- **Why:** Security tools builder, respects shipping
- **Pitch:** "Built a static analyzer specifically for AI-generated code - catches phantom packages that AI hallucinates. Different problem than traditional SAST."
- **Contact:** Twitter @haroonmeer

---

# TIER 2: Could Care

## AI Company Leaders

### Anysphere Team (Cursor)
- **Role:** Building Cursor (AI code editor)
- **Why:** They generate code - they need security
- **Pitch:** "SABS could integrate into Cursor's code generation to prune insecure paths during generation, not after. Happy to discuss integration or licensing."
- **Contact:** founders@cursor.so

### Amjad Masad
- **Role:** CEO, Replit
- **Why:** AI code generation is core to Replit
- **Pitch:** "SelfEngine/SABS makes code generation secure by design - integrates security verification into beam search. Could be valuable for Replit's AI features."
- **Contact:** Twitter @amasad

### Pieter Levels
- **Role:** Indie hacker, maker (12 startups)
- **Why:** Celebrates fast builders, has massive audience
- **Pitch:** "Shipped 3 AI safety products in 2 weeks with 25K total lines of code. Multi-agent prompt refinement, AI code scanner, and novel ML algorithm."
- **Contact:** Twitter @levelsio

---

## Tech Influencers

### ThePrimeagen
- **Role:** Netflix engineer, streamer, dev influencer
- **Why:** Cares about dev tools, has huge audience
- **Pitch:** "Built a CLI that catches when Copilot hallucinates fake npm packages. Demo would make great content."
- **Contact:** Twitter @ThePrimeagen

### Fireship (Jeff Delaney)
- **Role:** YouTube dev educator (2M+ subs)
- **Why:** Makes videos about new dev tools
- **Pitch:** "VibeGuard in 100 seconds? CLI that scans AI code for hallucinations, secrets, and insecure defaults - then auto-fixes."
- **Contact:** Twitter @fireship_dev

### Theo (t3.gg)
- **Role:** Former Twitch, T3 stack creator
- **Why:** Cares about dev experience, TypeScript
- **Pitch:** "Built Nexus - 4 AI agents refine your prompts collaboratively. Full TypeScript, Next.js, Supabase. The UI is actually good."
- **Contact:** Twitter @t3dotgg

---

## AI Newsletter Writers

### Matt Shumer
- **Role:** AI newsletter, active builder
- **Why:** Writes about AI tools weekly
- **Pitch:** "Three AI safety tools in one portfolio: Nexus (prompt refinement), VibeGuard (AI code scanner), SelfEngine (secure code gen research). Any interest in featuring?"
- **Contact:** Twitter @MattShumer_

### Zack Kass
- **Role:** Former OpenAI, AI thought leader
- **Why:** Writes about AI impact, has network
- **Pitch:** "Addressing AI safety at the tool level - built products that make AI-generated content safer. Practical safety, not theoretical."
- **Contact:** Twitter @zaborkhunt

---

# TIER 3: Worth a Shot

## Academic Researchers

### Percy Liang
- **Role:** Stanford HAI, HELM creator
- **Why:** Leads LLM evaluation research
- **Pitch:** "SABS paper explores security-aware beam search for code LLMs. Would love feedback from the HELM team."
- **Contact:** pliang@cs.stanford.edu

### Dawn Song
- **Role:** Berkeley, security + AI
- **Why:** Intersection of security and ML
- **Pitch:** "Combined my security and ML interests - SABS integrates static analysis into beam search for secure code generation."
- **Contact:** UC Berkeley faculty page

---

## Investors (Angels)

### Elad Gil
- **Role:** Angel investor, ex-Google, ex-Twitter
- **Why:** Invests in technical founders
- **Pitch:** "CS student, shipped 3 AI safety products with 25K lines of code. Looking for angel to help with GTM."
- **Contact:** Twitter @eloalgil

### Lachy Groom
- **Role:** Angel, ex-Stripe
- **Why:** Invests early, developer tools
- **Pitch:** "Built developer tools for AI safety: code scanner, prompt refinement, secure code gen. Seeking advice on GTM."
- **Contact:** Twitter @lfrith

### Josh Buckley
- **Role:** CEO Kick, angel investor
- **Why:** Invests in ambitious young founders
- **Pitch:** "20-year-old shipping AI safety tools. 3 products, 25K LOC, 2 weeks. Looking for advice/investment."
- **Contact:** Twitter @joshbuckley

---

# Outreach Templates

## Cold Twitter DM Template

```
Hey [Name],

I saw your work on [specific thing they did]. Really inspired me.

I'm a CS student who just shipped [Product] - [one sentence description].

[Link to demo/repo]

Would love your feedback if you have 2 mins.

- Rahul
```

## Cold Email Template

```
Subject: [Product Name] - [One-liner]

Hi [Name],

I'm Rahul, a CS student at Virginia Tech. I built [Product] because [problem].

[2 sentences on what it does]

It's live at [link] with [proof point - users, code quality, tests].

I'd love 15 minutes to get your feedback. Would [date] work?

Best,
Rahul
```

## LinkedIn Template

```
Hi [Name],

Your work on [specific thing] caught my attention.

I just shipped [Product] - [one sentence]. Given your background in [their expertise], I'd love your thoughts.

[Link]

Thanks for considering!
Rahul
```

---

# Outreach Tracker

| Name | Platform | Sent Date | Response | Follow-up |
|------|----------|-----------|----------|-----------|
| | | | | |
| | | | | |
| | | | | |

**Rule:** Follow up after 5 days if no response. Max 2 follow-ups.

---

# Priority Order

1. **This week:** Tier 1 AI Safety (Jan Leike, Anthropic team)
2. **Next week:** Tier 1 VCs (Nat Friedman, Guillermo Rauch)
3. **Week 3:** Tier 2 Influencers (ThePrimeagen, Fireship)
4. **Week 4:** Tier 3 Academics + Angels
