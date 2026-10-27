import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { READ_FIRST_HINT, withReadFirstHint } from '../src/routing/read-first-hint.js';
import { MessagesBody } from '../src/routing/messages-body.js';

describe('Read First D2 changes', () => {
  describe('a) withReadFirstHint placement', () => {
    it('appends to string content of last user message', () => {
      const bString = MessagesBody.parse({ model: 'x', messages: [{ role: 'user', content: 'hi' }], system: 'A' });
      const origString = structuredClone(bString);
      const res = withReadFirstHint(bString);
      assert.deepEqual(bString, origString, 'Input object must not be mutated');
      assert.equal(res.system, 'A', 'System unchanged');
      assert.deepEqual(res.messages[0]!.content, [
        { type: 'text', text: 'hi' },
        { type: 'text', text: '<system-reminder>\n' + READ_FIRST_HINT + '\n</system-reminder>' },
      ]);
    });

    it('appends to block content of last user message', () => {
      const bBlock = MessagesBody.parse({ model: 'x', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] });
      const origBlock = structuredClone(bBlock);
      const res = withReadFirstHint(bBlock);
      assert.deepEqual(bBlock, origBlock);
      assert.deepEqual(res.messages[0]!.content, [
        { type: 'text', text: 'hi' },
        { type: 'text', text: '<system-reminder>\n' + READ_FIRST_HINT + '\n</system-reminder>' },
      ]);
    });

    it('leaves non-user last message unchanged', () => {
      const bNon = MessagesBody.parse({
        model: 'x',
        messages: [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'hello' },
        ],
      });
      const orig = structuredClone(bNon);
      const res = withReadFirstHint(bNon);
      assert.deepEqual(res, orig);
    });
  });
});
