"""Create the short final PDF from completed results; no model calls."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import common as c

def main():
    summaries={n:c.read(c.OUT/n/'summary.json') for n in ('reasoning','rendering','confirmation','invariance')}
    if not all(summaries.values()): raise ValueError('All planned result summaries required')
    assert c.read(c.OUT/'final_integrity.json')['status']=='passed'
    r=summaries['reasoning']; v=summaries['rendering']; f=summaries['confirmation']; mirror=summaries['invariance']
    choice=c.read(c.OUT/'candidate.json'); ledger=c.read(c.OUT/'attempts.json'); spent=c.usage(ledger)['estimated_usd']+c.PRIOR_COST
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=22,leading=25,textColor=colors.HexColor('#142f47'),spaceAfter=12))
    styles.add(ParagraphStyle(name='ReportHead',fontName='Helvetica-Bold',fontSize=11.5,leading=15,textColor=colors.HexColor('#142f47'),spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9.5,leading=13.1,textColor=colors.HexColor('#283745'),spaceAfter=7))
    styles.add(ParagraphStyle(name='ReportSmall',fontName='Helvetica',fontSize=8.2,leading=11,textColor=colors.HexColor('#53616c'),spaceAfter=6))
    def p(text,style='ReportBody'): return Paragraph(text,styles[style])
    def table(rows,widths):
        values=[[p(str(cell),'ReportSmall') for cell in row] for row in rows]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef4')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#a3b5c4')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    test=f['primary'][0]; lo,hi=test['interval']
    if hi<0: headline='Luna had lower probability error than logistic regression on the confirmation sample.'
    elif lo>0: headline='Luna had higher probability error than logistic regression on the confirmation sample.'
    else: headline='A reliable win over logistic regression was not established.'
    score_rows=[['Method','Brier: lower is better','AUC: higher is better']]
    names={'candidate':f"Luna: {choice['effort']} reasoning, {choice['renderer']} diagram",'baseline_xg':'Logistic baseline: four inputs','baseline_image_xg':'Logistic without header input','sb_xg':'StatsBomb reference'}
    for key,name in names.items():
        s=f['scores'][key]; auc=f"{s['auc']:.3f}" if s['auc'] is not None else 'undefined'
        score_rows.append([name,f"{s['brier']:.5f}",auc])
    story=[p('Direct-image xG: findings','ReportTitle'),p('Research summary | Run started 29 October 2026','ReportSmall'),
           p('<b>'+headline+'</b>'),
           p(f"The independent check used <b>{f['shots']} shots, {f['goals']} goals and {f['matches']} matches</b>. Luna predicted xG directly from the shot diagram. No recalibration, statistical prediction head or adjustment changed its answers."),
           table(score_rows,[300,105,105]),Spacer(1,8),
           p(f"Primary Brier difference (Luna minus logistic): <b>{test['brier_difference']:+.5f}</b>. The 97.5% match-based interval is <b>[{lo:+.5f}, {hi:+.5f}]</b>. Negative values favor Luna; an interval crossing zero cannot establish a win.",'ReportSmall'),
           p(f"Average predicted chance: Luna {f['scores']['candidate']['mean_xg']:.1%}; logistic {f['scores']['baseline_xg']['mean_xg']:.1%}; actual scoring rate {f['goals']/f['shots']:.1%}. The predictions were left unchanged.",'ReportSmall'),
           p('What we tried','ReportHead'),
           table([['Experiment','Result'],
                  ['More reasoning: 50 new shots',f"Brier: low {r['scores']['low']['brier']:.4f}, medium {r['scores']['medium']['brier']:.4f}, high {r['scores']['high']['brier']:.4f}. The earlier small pilot benefit did not repeat reliably; low effort was selected."],
                  ['Less marker overlap: 40 development shots',f"Filled star {v['scores']['standard']['brier']:.4f}; hollow star {v['scores']['outline']['brier']:.4f}. No reliable benefit; the original renderer was selected."],
                  ['Independent check: 150 new shots',headline]], [165,345]),
           Spacer(1,8),p('The earlier 12-shot pilot contained only two goals. One scored shot contributed about 62% of its apparent gain. It was a useful feasibility signal, not proof of a better method.','ReportSmall'),
           p('Brier measures probability error; AUC measures ranking. A model can improve one without improving the other. Both are reported.','ReportSmall'),PageBreak(),
           p('What changed - and what did not','ReportTitle'),
           Image(str(c.OUT/'renderer_preview/comparison.png'),width=440,height=189),
           p('Same development scene, same player positions. The hollow star reveals part of an overlapping defender. Better visibility alone did not establish better xG predictions.','ReportSmall'),
           p('Consistency check','ReportHead'),
           p(f"On {mirror['shots']} development scenes, repeating the identical image changed xG by an average of <b>{100*mirror['mean_repeat_change']:.2f} percentage points</b>. Mirroring the image changed it by <b>{100*mirror['mean_mirror_change']:.2f} points</b>. The mirror-minus-repeat 95% interval was [{100*mirror['mirror_minus_repeat_interval_95'][0]:+.2f}, {100*mirror['mirror_minus_repeat_interval_95'][1]:+.2f}] points. This is a sensitivity diagnostic, not extra independent goal-labelled evidence."),
           p('Why the comparison is credible','ReportHead'),
           p('The source stayed StatsBomb Open Data. We excluded 264 previously used match IDs and reserved separate AFCON 2023 match groups before development. The final method was frozen before confirmation scores were viewed. Outcomes, numerical coordinates, body-part labels and comparator xG were never sent to Luna.'),
           p('The four-feature logistic baseline uses distance, angle, a header flag and opponent count, learned from 3,068 World Cup shots. The header flag is information absent from the image. A body-part-free comparator is therefore shown separately. Both numerical models receive recorded coordinates. One tournament and finite goal counts limit broader conclusions.','ReportSmall'),
           p('Cost and stopping decision','ReportHead'),
           p(f"<b>Estimated API spending: ${spent:.4f}</b>, including the earlier pilot. The new campaign completed {len(ledger)} generation calls; all used GPT-6 Luna. The working cap was $2, below the approximately $10 account limit. These are usage-derived estimates, not a balance or billing check."),
           p('The planned reasoning, rendering, independent-check and consistency studies are complete. Further prompt searching on the same confirmation outcomes would weaken the evidence. There is no need to spend the remaining budget.'),
           p('Detailed methods, uncertainty intervals, per-shot predictions, raw responses and integrity checks are saved beside this PDF in FINAL_REPORT.md and the stage folders.','ReportSmall'),
           p('Sources: <link href="https://github.com/hudl/open-data" color="#245c86">StatsBomb Open Data</link> | <link href="https://developers.openai.com/api/docs/models/gpt-6-luna" color="#245c86">Official Luna rates and settings</link>','ReportSmall')]
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFillColor(colors.HexColor('#657583')); canvas.setFont('Helvetica',8)
        canvas.drawString(42,25,'Direct image-only VLM xG | Raw predictions | No recalibration')
        canvas.drawRightString(570,25,str(doc.page)); canvas.restoreState()
    target=c.OUT/'RESEARCH_SUMMARY.pdf'
    SimpleDocTemplate(str(target),pagesize=letter,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=42,
                      title='Direct-image xG research findings',author='xG Research').build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created readable summary:',target)
if __name__=='__main__': main()


