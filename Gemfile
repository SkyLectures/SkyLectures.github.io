source "https://rubygems.org"

gem "jekyll", "~> 4.3"

# If you're using the MathJax math engine instead, free to remove the line below:
gem "kramdown-math-katex"

# A JavaScript runtime for Ruby that helps with running the katex gem above.
gem "duktape"

# Required for `jekyll serve` in Ruby 3
gem "webrick"

# Uncomment when using the `--lsi` option for `jekyll build`
# gem "classifier-reborn"

group :jekyll_plugins do
  gem "jekyll-default-layout"
  gem "jekyll-feed"
  gem "jekyll-optional-front-matter"
  gem "jekyll-paginate"
  gem "jekyll-readme-index"
  gem "jekyll-redirect-from"
  gem "jekyll-relative-links"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-titles-from-headings"
  gem "jekyll-include-cache"

  # Non-Github Pages plugins:
  gem "jekyll-last-modified-at"
  gem "jekyll-compose"
end

gem 'wdm' if Gem.win_platform?
gem "tzinfo-data" if Gem.win_platform?

