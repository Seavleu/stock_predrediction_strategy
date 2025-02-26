def rank_stocks(data):
    """Rank top 5 stocks based on technical indicators and market performance."""
    data['score'] = (data['momentum_score'] + data['volume_score']) / 2
    top_5 = data.sort_values(by='score', ascending=False).head(5)
    return top_5
